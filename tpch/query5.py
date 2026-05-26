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

# almost TPCH query 5
QUERY5_SQL = """
select
        n_name,
        sum(l_extendedprice * (1 - l_discount)) as revenue
from
        customer,
        orders,
        lineitem,
        supplier,
        nation
where
        c_custkey = o_custkey
        and l_orderkey = o_orderkey
        and l_suppkey = s_suppkey
        and c_nationkey = s_nationkey
        and s_nationkey = n_nationkey
        and n_regionkey = 2
        and o_orderdate >= date '1994-01-01'
        and o_orderdate < date '1994-01-01' + interval '1' year
group by
        n_name
order by
        n_name
"""

def query5_agg(sorted_data,groups_unique,group_idx):
    # sum(l_extendedprice * (1 - l_discount)) as revenue
    discount_factor = (100 - sorted_data[1])
    revenue_per_item = sorted_data[0] * discount_factor
    results = torch.zeros(groups_unique.shape[0],dtype=revenue_per_item.dtype,device=sorted_data[0].get_device()).scatter_reduce(0, group_idx, revenue_per_item, reduce="sum")
    return results

def query5_agg_combine(sorted_data,groups_unique,group_idx):
    results = torch.zeros(groups_unique.shape[0],dtype=sorted_data[1].dtype,device=sorted_data[0].get_device()).scatter_reduce(0, group_idx, sorted_data[1], reduce="sum")
    return results


def query5_opt_probe_pipeline(last_join, aggregation, joined_tuples):
    # after customer join:
    # [ 0,           1,             2,           3,          4,            5,           6,          7,                8,           9]
    # ["c_custkey", "c_nationkey","o_orderkey","o_custkey","o_orderdate","l_orderkey","l_suppkey","l_extendedprice","l_discount","l_shipdate"]
    # ["o_orderkey","c_nationkey","l_orderkey","l_suppkey","l_extendedprice","l_discount","l_shipdate"]
    # after supplier join +2
    # after nation join +3
    for final_data in last_join.probe(joined_tuples, 3): # join (NxS) using l_suppkey
        mask = torch.eq(final_data[4], final_data[6]) # s_nationkey = c_nationkey
        # filtered = [torch.masked_select(col, mask) for col in final_data]
        # project to ["n_name","l_extendedprice","l_discount"]
        filtered = [torch.masked_select(final_data[col_idx], mask) for col_idx in [1,9,10]]
        aggregation(filtered)


def run_query5_on_parquet(iteration, store, target_device, scale_factor):
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
                                                ("nation", f"{base_folder}/nation.parquet", storage_threads,
                                                    ["n_nationkey", "n_name", "n_regionkey"],
                                                    TableFilter({
                                                        # r_name = r_regionkey = n_regionkey = ASIA
                                                        2: [FilterCondition(FilterOp.EQ, string_to_categoryidx("r_name","ASIA"))]
                                                    })
                                                ),

                                                ("supplier", f"{base_folder}/supplier.parquet", storage_threads,
                                                    ["s_suppkey", "s_nationkey"], None
                                                ),

                                                ("customer", f"{base_folder}/customer.parquet", storage_threads,
                                                    ["c_custkey","c_nationkey"], None
                                                ),

                                                ("orders", f"{base_folder}/orders.parquet", storage_threads,
                                                    ["o_orderkey", "o_custkey", "o_orderdate"],
                                                    TableFilter({
                                                        2: [FilterCondition(FilterOp.GE, datetime.date(1994, 1, 1)),
                                                            FilterCondition(FilterOp.LT, datetime.date(1995, 1, 1))]  # o_orderdate
                                                    })
                                                ),

                                                ("lineitem", f"{base_folder}/lineitem.parquet", storage_threads,
                                                    ["l_orderkey","l_suppkey","l_extendedprice","l_discount"],None
                                                )
                                        ])

    nation_reader, supplier_reader, customer_reader, orders_reader, lineitem_reader = reader_wrapper.readers
    if os.environ["HASH_SIZE_NAIVE"] == "True":
        nation_supplier_hash_init = 200
        cust_order_hash_init = 160000*scale_factor
        last_join_hash_init = scale_factor*11000
        order_lineitem_hash_init = (scale_factor * 1000000) // WORLD_SIZE
    else:
        nation_supplier_hash_init = util.next_prime(200)
        cust_order_hash_init = util.next_prime(160000*scale_factor)
        last_join_hash_init = util.next_prime(scale_factor*11000)
        order_lineitem_hash_init = util.next_prime((scale_factor * 1000000) // WORLD_SIZE)

    torch.cuda.empty_cache()
    import gc
    gc.collect()

    customer_orders_join = SingleNodeJoin(target_device, cust_order_hash_init, customer_reader.get_fifo(),
                                            right_child=orders_reader.get_fifo(), right_col_idx=1)

    nation_supplier_join = SingleNodeJoin(target_device, nation_supplier_hash_init, nation_reader.get_fifo(),
                                            right_child=supplier_reader.get_fifo(), right_col_idx=1)
    # this last one (NxS) x ((CxO)xL) is probed last
    last_join = SingleNodeJoin(target_device, last_join_hash_init, nation_supplier_join)

    shuffle_left_child = customer_orders_join
    shuffle_right_child = lineitem_reader.get_fifo()

    if (tb_size := os.environ.get("SHUFFLE_BUFFER_LEFT")) is not None:
        print(f"# Adding tensor buffer for left join side (size {int(tb_size)})")
        shuffle_left_child = TensorBuffer(customer_orders_join, int(tb_size))
    if (tb_size := os.environ.get("SHUFFLE_BUFFER_RIGHT")) is not None:
        print(f"# Adding tensor buffer for right join side (size {int(tb_size)})")
        shuffle_right_child = TensorBuffer(lineitem_reader.get_fifo(), int(tb_size))

    # build side: ["c_custkey","c_nationkey","o_orderkey", "o_custkey", "o_orderdate"]
    co_lineitem = SHUFFLE_MODULE_TYPE(target_device, order_lineitem_hash_init, shuffle_left_child, 2,
                                         shuffle_right_child, 0, store, store_prefix="", project_build_side=[2,1])
    # before aggregation ["n_name","l_extendedprice","l_discount"]
    # query5_aggregation = AggregationPostCombine(query5_agg, query5_agg_combine, [1], projection=[11,12])
    query5_aggregation = AggregationPostCombine(query5_agg, query5_agg_combine, [0], projection=[1,2])
    probe_pipeline = partial(query5_opt_probe_pipeline, last_join, query5_aggregation)

    ######################################################################################
    def tmp_function():
        build_start = time.perf_counter_ns()
        nation_supplier_join.build(nation_reader.get_fifo(), 0)     # build on n_nationkey
        customer_orders_join.build(customer_reader.get_fifo(), 0)   # build on c_custkey
        nation_supplier_join.finish()
        last_join.build(nation_supplier_join, 3)                    # build on s_suppkey
        customer_orders_join.finish()
        last_join.finish()
        build_end = time.perf_counter_ns()
        # print(f"Single Node build sides took {(build_end-build_start)/1e6:.6f}ms")

        result = co_lineitem(probe_pipeline=probe_pipeline)
        agg_finalize_start = time.perf_counter_ns()
        query5_aggregation.finalize()
        torch.cuda.synchronize(device=target_device)

        result["time_finalize_agg_ns"] = time.perf_counter_ns() - agg_finalize_start
        return result

    result_str = reader_wrapper.run(tmp_function)
    print(f"# customer_orders_join tuples      : {customer_orders_join.num_join_tuples}")
    print(f"# nation_supplier_join tuples      : {nation_supplier_join.num_join_tuples}")
    print(f"# last_join                        : {last_join.num_join_tuples}")
    if iteration == 0:
        print(f"query,iteration,algo,{reader_wrapper.header_str}")
    print(f"query5,{iteration},{co_lineitem.name()},{result_str}")
    ######################################################################################

    if os.environ["SHUFFLE_PROBE_TYPE"] != "FULL":
        return

    ######################################################################################
    groups, aggs = query5_aggregation.get_results()

    torch.set_printoptions(linewidth=200,threshold=1000,precision=4,sci_mode=False)
    print(f"# [Local] Result groups: {groups.shape=}")
    print("# n_name      : ",[categoryidx_to_string("n_name",x) for x in groups[:,0].tolist()])
    print("# revenue     : ",aggs)

    if os.environ.get("SYNC_RESULT", "False") == "True" and WORLD_SIZE > 1:
        all_groups = [None] * WORLD_SIZE
        all_aggs = [None] * WORLD_SIZE
        dist.all_gather_object(all_groups, groups)
        dist.all_gather_object(all_aggs, aggs)
        if dist.get_rank() == 0: # global results in node0
            non_empty_groups = [g for g in all_groups if g.shape[0] > 0]
            non_empty_aggs = [a for a in all_aggs if a.shape[0] > 0]
            if non_empty_groups:
                combined_groups = torch.cat(non_empty_groups, dim=0)
                combined_aggs = torch.cat(non_empty_aggs, dim=0)
                unique_groups, group_indices = torch.unique(combined_groups, dim=0, return_inverse=True)
                final_aggs = torch.zeros(unique_groups.shape[0], dtype=combined_aggs.dtype, device=combined_aggs.device)
                final_aggs.scatter_reduce_(0, group_indices, combined_aggs, reduce="sum")
                print(f"# [Global] Result groups: {unique_groups.shape=}")
                print("# n_nationkey : ",unique_groups[:,0])
                # print("# n_regionkey : ",unique_groups[:,1])
                print("# revenue     : ",final_aggs)

    torch.set_printoptions()