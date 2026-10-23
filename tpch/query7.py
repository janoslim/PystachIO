import sys
import os
import time
import datetime
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
import pandas as pd
from functools import partial
import torch
import torch.distributed as dist

from helpers import util
from modules.filter_module import FilterCondition, FilterOp, TableFilter
from modules.single_thread_naive import ShuffleWithJoinNaive
from modules.single_thread_shuffle_join_multistream import ShuffleWithJoin_MultiStream
from modules.tensor_buffer import TensorBuffer
from modules.query_wrapper import ReaderWrapper
from modules.single_node_join import SingleNodeJoin
from modules.aggregation_module import Aggregation,AggregationPostCombine
from .tpch import get_orders,get_customer,get_lineitem
from .dict import *

QUERY7_SQL = """
select
        supp_nation,
        cust_nation,
        l_year,
        sum(volume) as revenue
from
        (
                select
                        n1.n_name as supp_nation,
                        n2.n_name as cust_nation,
                        extract(year from l_shipdate) as l_year,
                        l_extendedprice * (1 - l_discount) as volume
                from
                        supplier,
                        lineitem,
                        orders,
                        customer,
                        nation n1,
                        nation n2
                where
                        s_suppkey = l_suppkey
                        and o_orderkey = l_orderkey
                        and c_custkey = o_custkey
                        and s_nationkey = n1.n_nationkey
                        and c_nationkey = n2.n_nationkey
                        and (
                                (n1.n_name = 'FRANCE' and n2.n_name = 'GERMANY')
                                or (n1.n_name = 'GERMANY' and n2.n_name = 'FRANCE')
                        )
                        and l_shipdate between date '1995-01-01' and date '1996-12-31'
        ) as shipping
group by
        supp_nation,
        cust_nation,
        l_year
order by
        supp_nation,
        cust_nation,
        l_year
"""

def query7_agg(sorted_data,groups_unique,group_idx):

    # s_nationkey, c_nationkey, year, l_extendedprice, l_discount
    # l_extendedprice * (1 - l_discount)
    revenue = sorted_data[3] * (100 - sorted_data[4])

    results = torch.zeros(groups_unique.shape[0],dtype=revenue.dtype,device=sorted_data[0].get_device()).scatter_reduce(0, group_idx, revenue, reduce="sum")
    return results

def query7_agg_combine(sorted_data,groups_unique,group_idx):
    results = torch.zeros(groups_unique.shape[0],dtype=sorted_data[1].dtype,device=sorted_data[0].get_device()).scatter_reduce(0, group_idx, sorted_data[1], reduce="sum")
    return results


def query7_probe_pipeline(last_join, aggregation, joined_tuples):
    # after shuffle:
    # [ 0,            1,           2,          3,            4,           5,          6,                7,           8]
    # ["o_orderkey", "o_custkey", "s_suppkey","s_nationkey","l_orderkey","l_suppkey","l_extendedprice","l_discount","l_shipdate"]
    # after last_join prepended: ["c_custkey","c_nationkey"]

    for final_data in last_join.probe(joined_tuples, 1): # join c using o_custkey
        mask = torch.ne(final_data[5], final_data[1]) # s_nationkey != c_nationkey
        filtered = [torch.masked_select(final_data[col_idx], mask) for col_idx in [5,1,10,8,9]]
        # s_nationkey, c_nationkey, year, l_extendedprice, l_discount
        # not exactly year, but close enough
        # year = torch.div(final_data[10], 365, rounding_mode="floor") + 1970
        # aggregation([final_data[5], final_data[1], year, final_data[8], final_data[9]])
        filtered[2] = torch.div(filtered[2], 365, rounding_mode="floor") + 1970
        aggregation(filtered)


def run_query7_on_parquet(iteration, store, target_device, scale_factor):
    WORLD_SIZE = int(os.environ['WORLD_SIZE'])
    if os.environ["SHUFFLE_TYPE"] == "BLOCKING":
        SHUFFLE_MODULE_TYPE = ShuffleWithJoinNaive
    elif "OVERLAPPED" in os.environ["SHUFFLE_TYPE"]:
        SHUFFLE_MODULE_TYPE = ShuffleWithJoin_MultiStream
    else:
        print(f"Error, unknown SHUFFLE_TYPE={os.environ["SHUFFLE_TYPE"]}")
        return
    storage_threads = int(os.environ["STORAGE_THREADS"])

    base_folder = f"{os.environ['STORAGE_BASE_PATH']}/tpch-sf{scale_factor}/"

    reader_wrapper = ReaderWrapper(store, target_device,
                                         # name, filepath, threads, cols, filters
                                table_infos= [  
                                                ("customer", f"{base_folder}/customer.parquet", storage_threads,
                                                    ["c_custkey","c_nationkey"],
                                                    TableFilter({
                                                        1: [FilterCondition(FilterOp.II, torch.tensor([6,7],device=target_device))]  # c_nationkey
                                                    })
                                                ),

                                                ("supplier", f"{base_folder}/supplier.parquet", storage_threads,
                                                    ["s_suppkey", "s_nationkey"],
                                                    TableFilter({
                                                        1: [FilterCondition(FilterOp.II, torch.tensor([6,7],device=target_device))]  # s_nationkey
                                                    })
                                                ),

                                                ("orders", f"{base_folder}/orders.parquet", storage_threads,
                                                    ["o_orderkey", "o_custkey"], None
                                                ),

                                                ("lineitem", f"{base_folder}/lineitem.parquet", storage_threads,
                                                    ["l_orderkey","l_suppkey","l_extendedprice","l_discount","l_shipdate"],
                                                    TableFilter({
                                                        4: [FilterCondition(FilterOp.GE, datetime.date(1995, 1, 1)),
                                                            FilterCondition(FilterOp.LE, datetime.date(1996, 12, 31))]  # l_shipdate
                                                    })
                                                )
                                        ])

    customer_reader, supplier_reader, orders_reader, lineitem_reader = reader_wrapper.readers
    if os.environ["HASH_SIZE_NAIVE"] == "True":
        supplier_lineitem_hash_init = 8000*scale_factor
        last_join_hash_init = scale_factor*14000
        order_lineitem_hash_init = (scale_factor * 1000000) // WORLD_SIZE
    else:
        supplier_lineitem_hash_init = util.next_prime(8000*scale_factor)
        last_join_hash_init = util.next_prime(scale_factor*14000)
        order_lineitem_hash_init = util.next_prime((scale_factor * 1000000) // WORLD_SIZE)

    torch.cuda.empty_cache()
    import gc
    gc.collect()

    supplier_lineitem_join = SingleNodeJoin(target_device, supplier_lineitem_hash_init, supplier_reader.get_fifo(),
                                            right_child=lineitem_reader.get_fifo(), right_col_idx=1)
    # this last one is probed last
    last_join = SingleNodeJoin(target_device, last_join_hash_init, customer_reader.get_fifo())

    shuffle_left_child = orders_reader.get_fifo()
    shuffle_right_child = supplier_lineitem_join

    if (tb_size := os.environ.get("SHUFFLE_BUFFER_LEFT")) is not None:
        print(f"# Adding tensor buffer for left join side (size {int(tb_size)})")
        shuffle_left_child = TensorBuffer(orders_reader.get_fifo(), int(tb_size))
    if (tb_size := os.environ.get("SHUFFLE_BUFFER_RIGHT")) is not None:
        print(f"# Adding tensor buffer for right join side (size {int(tb_size)})")
        shuffle_right_child = TensorBuffer(supplier_lineitem_join, int(tb_size))

    # build side: ["o_orderkey", "o_custkey"]
    # probe side: ["s_suppkey","s_nationkey","l_orderkey","l_suppkey","l_extendedprice","l_discount","l_shipdate"]
    shuffle_join = SHUFFLE_MODULE_TYPE(target_device, order_lineitem_hash_init, shuffle_left_child, 0,
                                         shuffle_right_child, 2, store, store_prefix="")

    # projection done in query7_probe_pipeline
    query7_aggregation = AggregationPostCombine(query7_agg, query7_agg_combine, [0,1,2])
    probe_pipeline = partial(query7_probe_pipeline, last_join, query7_aggregation)

    ######################################################################################
    def tmp_function():
        build_start = time.perf_counter_ns()
        last_join.build(customer_reader.get_fifo(), 0)                  # build on c_custkey
        supplier_lineitem_join.build(supplier_reader.get_fifo(), 0)     # build on s_suppkey

        last_join.finish()
        supplier_lineitem_join.finish()

        build_end = time.perf_counter_ns()
        # print(f"Single Node build sides took {(build_end-build_start)/1e6:.6f}ms")

        result = shuffle_join(probe_pipeline=probe_pipeline)
        agg_finalize_start = time.perf_counter_ns()
        query7_aggregation.finalize()
        torch.cuda.synchronize(device=target_device)

        result["time_finalize_agg_ns"] = time.perf_counter_ns() - agg_finalize_start
        return result

    result_str = reader_wrapper.run(tmp_function)
    print(f"# last_join tuples (build -> out)               : {last_join.build_side_cols[0].shape[0]} -> {last_join.num_join_tuples}")
    print(f"# supplier_lineitem_join tuples (build -> out)      : {supplier_lineitem_join.build_side_cols[0].shape[0]} -> {supplier_lineitem_join.num_join_tuples}")

    if iteration == 0:
        print(f"query,iteration,algo,{reader_wrapper.header_str}")
    print(f"query7,{iteration},{shuffle_join.name()},{result_str}")
    ######################################################################################

    if os.environ["SHUFFLE_PROBE_TYPE"] != "FULL":
        return

    ######################################################################################
    groups, aggs = query7_aggregation.get_results()
    from helpers.flowcsd_observe import dump_result
    dump_result({"groups": groups, "aggs": aggs})

    torch.set_printoptions(linewidth=200,threshold=1000,precision=4,sci_mode=False)
    print(f"# [Local] Result groups: {groups.shape=}")
    print("# s_nationkey    : ",groups[:,0])
    print("# c_nationkey    : ",groups[:,1])
    print("# year           : ",groups[:,2])
    print("# revenue          ",aggs)

    # if os.environ.get("SYNC_RESULT", "False") == "True" and WORLD_SIZE > 1:
    #     all_groups = [None] * WORLD_SIZE
    #     all_aggs = [None] * WORLD_SIZE
    #     dist.all_gather_object(all_groups, groups)
    #     dist.all_gather_object(all_aggs, aggs)
    #     if dist.get_rank() == 0: # global results in node0
    #         non_empty_groups = [g for g in all_groups if g.shape[0] > 0]
    #         non_empty_aggs = [a for a in all_aggs if a.shape[0] > 0]
    #         if non_empty_groups:
    #             combined_groups = torch.cat(non_empty_groups, dim=0)
    #             combined_aggs = torch.cat(non_empty_aggs, dim=0)
    #             unique_groups, group_indices = torch.unique(combined_groups, dim=0, return_inverse=True)
    #             final_aggs = torch.zeros(unique_groups.shape[0], dtype=combined_aggs.dtype, device=combined_aggs.device)
    #             final_aggs.scatter_reduce_(0, group_indices, combined_aggs, reduce="sum")
    #             print(f"# [Global] Result groups: {unique_groups.shape=}")
    #             print("# n_nationkey : ",unique_groups[:,0])
    #             # print("# n_regionkey : ",unique_groups[:,1])
    #             print("# revenue     : ",final_aggs)

    # torch.set_printoptions()