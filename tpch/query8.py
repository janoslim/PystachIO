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

QUERY8_SQL = """
select
        o_year,
        sum(case
                when nation = 'BRAZIL' then volume
                else 0
        end) / sum(volume) as mkt_share
from
    (
        select
                extract(year from o_orderdate) as o_year,
                l_extendedprice * (1 - l_discount) as volume,
                n2.n_name as nation
        from
                part,
                supplier,
                lineitem,
                orders,
                customer,
                nation n1,
                nation n2,
                region
        where
                p_partkey = l_partkey
                and s_suppkey = l_suppkey
                and l_orderkey = o_orderkey
                and o_custkey = c_custkey
                and c_nationkey = n1.n_nationkey
                and n1.n_regionkey = r_regionkey
                and r_name = 'AMERICA'
                and s_nationkey = n2.n_nationkey
                and o_orderdate between date '1995-01-01' and date '1996-12-31'
                and p_type = 'ECONOMY ANODIZED STEEL'
    ) as all_nations
group by
        o_year
order by
        o_year
"""

def query8_agg(sorted_data,groups_unique,group_idx):

    # ["year","s_nationkey","l_extendedprice","l_discount"]
    # l_extendedprice * (1 - l_discount)
    volume = sorted_data[2] * (100 - sorted_data[3])
    brazil_volume = volume * torch.eq(sorted_data[1], string_to_categoryidx("n_name","BRAZIL"))

    results = torch.zeros((groups_unique.shape[0], 2),dtype=torch.int64,device=sorted_data[0].get_device())
    results[:,0].scatter_reduce_(0, group_idx, volume, reduce="sum") # sum_volume
    results[:,1].scatter_reduce_(0, group_idx, brazil_volume, reduce="sum")
    return results

def query8_agg_combine(sorted_data,groups_unique,group_idx):
    results = torch.zeros((groups_unique.shape[0],2),dtype=torch.int64,device=sorted_data[0].get_device())

    results[:,0].scatter_reduce_(0, group_idx, sorted_data[1][:,0], reduce="sum")
    results[:,1].scatter_reduce_(0, group_idx, sorted_data[1][:,1], reduce="sum")
    return results


def query8_probe_pipeline(last_join, aggregation, joined_tuples):
    # after customer join:
    # [ 0,             1,             2,          3,            4,            5,          6,       7,           8,          9,          10,               11]
    # ["n_nationkey", "n_regionkey", "c_custkey","c_nationkey","o_orderkey", "o_custkey", "o_orderdate","p_partkey","p_type","l_orderkey","l_partkey","l_suppkey","l_extendedprice","l_discount"]

    for final_data in last_join.probe(joined_tuples, 11): # join (NxS) using l_suppkey
        # year, s_nationkey, l_extendedprice, l_discount
        # not exactly year, but close enough
        year = torch.div(final_data[8], 365, rounding_mode="floor") + 1970
        aggregation([year, final_data[1], final_data[12], final_data[13]])


def run_query8_on_parquet(iteration, store, target_device, scale_factor):
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
                                                ("supplier", f"{base_folder}/supplier.parquet", storage_threads,
                                                    ["s_suppkey", "s_nationkey"], None
                                                ),

                                                ("nation", f"{base_folder}/nation.parquet", storage_threads,
                                                    ["n_nationkey", "n_regionkey"],
                                                    TableFilter({
                                                        # r_name = r_regionkey = n_regionkey = AMERICA
                                                        1: [FilterCondition(FilterOp.EQ, string_to_categoryidx("r_name","AMERICA"))]
                                                    })
                                                ),

                                                ("part", f"{base_folder}/part.parquet", storage_threads,
                                                    ["p_partkey","p_type"],
                                                    TableFilter({
                                                                1: [FilterCondition(FilterOp.EQ, string_to_categoryidx("p_type","ECONOMY ANODIZED STEEL"))]
                                                                })
                                                ),

                                                ("customer", f"{base_folder}/customer.parquet", storage_threads,
                                                    ["c_custkey","c_nationkey"], None
                                                ),

                                                ("orders", f"{base_folder}/orders.parquet", storage_threads,
                                                    ["o_orderkey", "o_custkey", "o_orderdate"],
                                                    TableFilter({
                                                        2: [FilterCondition(FilterOp.GE, datetime.date(1995, 1, 1)),
                                                            FilterCondition(FilterOp.LE, datetime.date(1996, 12, 31))]  # o_orderdate
                                                    })
                                                ),

                                                ("lineitem", f"{base_folder}/lineitem.parquet", storage_threads,
                                                    ["l_orderkey","l_partkey","l_suppkey","l_extendedprice","l_discount"],None
                                                )
                                        ])

    supplier_reader, nation_reader, part_reader, customer_reader, orders_reader, lineitem_reader = reader_wrapper.readers
    if os.environ["HASH_SIZE_NAIVE"] == "True":
        cust_order_hash_init = 160000*scale_factor
        part_lineitem_hash_init = 3000*scale_factor
        last_join_hash_init = scale_factor*12000
        order_lineitem_hash_init = (scale_factor * 1000000) // WORLD_SIZE
    else:
        cust_order_hash_init = util.next_prime(160000*scale_factor)
        part_lineitem_hash_init = util.next_prime(3000*scale_factor)
        last_join_hash_init = util.next_prime(scale_factor*12000)
        order_lineitem_hash_init = util.next_prime((scale_factor * 1000000) // WORLD_SIZE)

    torch.cuda.empty_cache()
    import gc
    gc.collect()

    nation_customer_join = SingleNodeJoin(target_device, 20, nation_reader.get_fifo(),
                                            right_child=customer_reader.get_fifo(), right_col_idx=1)
    customer_orders_join = SingleNodeJoin(target_device, cust_order_hash_init, nation_customer_join,
                                            right_child=orders_reader.get_fifo(), right_col_idx=1)
    part_lineitem_join = SingleNodeJoin(target_device, part_lineitem_hash_init, part_reader.get_fifo(),
                                            right_child=lineitem_reader.get_fifo(), right_col_idx=1)
    # this last one S x ((CxO)x(PxL)) is probed last
    last_join = SingleNodeJoin(target_device, last_join_hash_init, supplier_reader.get_fifo())

    shuffle_left_child = customer_orders_join
    shuffle_right_child = part_lineitem_join

    if (tb_size := os.environ.get("SHUFFLE_BUFFER_LEFT")) is not None:
        print(f"# Adding tensor buffer for left join side (size {int(tb_size)})")
        shuffle_left_child = TensorBuffer(customer_orders_join, int(tb_size))
    if (tb_size := os.environ.get("SHUFFLE_BUFFER_RIGHT")) is not None:
        print(f"# Adding tensor buffer for right join side (size {int(tb_size)})")
        shuffle_right_child = TensorBuffer(part_lineitem_join, int(tb_size))

    # build side: ["n_nationkey", "n_regionkey", "c_custkey","c_nationkey","o_orderkey", "o_custkey", "o_orderdate"]
    # probe side: ["p_partkey","p_type","l_orderkey","l_partkey","l_suppkey","l_extendedprice","l_discount"]
    shuffle_join = SHUFFLE_MODULE_TYPE(target_device, order_lineitem_hash_init, shuffle_left_child, 4,
                                         shuffle_right_child, 2, store, store_prefix="")

    # projection done in query8_probe_pipeline
    query8_aggregation = AggregationPostCombine(query8_agg, query8_agg_combine, [0])
    probe_pipeline = partial(query8_probe_pipeline, last_join, query8_aggregation)

    ######################################################################################
    def tmp_function():
        build_start = time.perf_counter_ns()
        last_join.build(supplier_reader.get_fifo(), 0)              # build on s_suppkey
        nation_customer_join.build(nation_reader.get_fifo(), 0)     # build on n_nationkey
        part_lineitem_join.build(part_reader.get_fifo(), 0)         # build on p_partkey
        nation_customer_join.finish()
        customer_orders_join.build(nation_customer_join, 2)         # build on c_custkey

        last_join.finish()
        part_lineitem_join.finish()
        customer_orders_join.finish()

        build_end = time.perf_counter_ns()
        # print(f"Single Node build sides took {(build_end-build_start)/1e6:.6f}ms")


        result = shuffle_join(probe_pipeline=probe_pipeline)
        agg_finalize_start = time.perf_counter_ns()
        query8_aggregation.finalize()
        torch.cuda.synchronize(device=target_device)

        result["time_finalize_agg_ns"] = time.perf_counter_ns() - agg_finalize_start
        return result

    result_str = reader_wrapper.run(tmp_function)
    print(f"# last_join tuples (build -> out)               : {last_join.build_side_cols[0].shape[0]} -> {last_join.num_join_tuples}")
    print(f"# nation_customer_join tuples (build -> out)    : {nation_customer_join.build_side_cols[0].shape[0]} -> {nation_customer_join.num_join_tuples}")
    print(f"# part_lineitem_join tuples (build -> out)      : {part_lineitem_join.build_side_cols[0].shape[0]} -> {part_lineitem_join.num_join_tuples}")
    print(f"# customer_orders_join tuples (build -> out)    : {customer_orders_join.build_side_cols[0].shape[0]} -> {customer_orders_join.num_join_tuples}")

    if iteration == 0:
        print(f"query,iteration,algo,{reader_wrapper.header_str}")
    print(f"query8,{iteration},{shuffle_join.name()},{result_str}")
    ######################################################################################

    if os.environ["SHUFFLE_PROBE_TYPE"] != "FULL":
        return

    ######################################################################################
    groups, aggs = query8_aggregation.get_results()
    from helpers.flowcsd_observe import dump_result
    dump_result({"groups": groups, "aggs": aggs})

    torch.set_printoptions(linewidth=200,threshold=1000,precision=4,sci_mode=False)
    print(f"# [Local] Result groups: {groups.shape=}")
    print("# year               : ",groups[:,0])
    print("# sum_volume         : ",aggs[:,0])
    print("# brazil_volume      : ",aggs[:,1])
    print("# brazil iteration   : ",aggs[:,1]/aggs[:,0])

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