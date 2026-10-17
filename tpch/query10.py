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
from helpers.flowcsd_observe import dump_result
from modules.filter_module import FilterCondition, FilterOp, TableFilter
from modules.single_thread_naive import ShuffleWithJoinNaive
from modules.single_thread_shuffle_join_multistream import ShuffleWithJoin_MultiStream
from modules.grace_join import GraceHashJoin

from modules.query_wrapper import ReaderWrapper
from modules.join_wrapper_module import join_wrapper
from modules.aggregation_module import Aggregation,AggregationPostCombine
from modules.pipeline_breaker import PipelineBreaker
from modules.tensor_buffer import TensorBuffer
from modules.single_node_join import SingleNodeJoin
from .tpch import get_orders,get_customer,get_lineitem
from .dict import *

# almost query 10
#   for running:           l_returnflag = 2
QUERY10_SQL = """
select
        c_custkey,
        c_nationkey,
        c_acctbal,
        sum(l_extendedprice * (1 - l_discount)) as revenue
from
        customer,
        orders,
        lineitem
where
        c_custkey = o_custkey
        and l_orderkey = o_orderkey
        and o_orderdate >= date '1993-10-01'
        and o_orderdate < date '1993-10-01' + interval '3' month
        and l_returnflag = 'R'
group by
        c_custkey,
        c_nationkey,
        c_acctbal
order by
        c_custkey,
        c_nationkey,
        c_acctbal
"""

def query10_agg(sorted_data,groups_unique,group_idx):
    # sum(l_extendedprice * (1 - l_discount)) as revenue
    discount_factor = (100 - sorted_data[1])
    revenue_per_item = sorted_data[0] * discount_factor
    results = torch.zeros(groups_unique.shape[0],dtype=revenue_per_item.dtype,device=sorted_data[0].get_device()).scatter_reduce(0, group_idx, revenue_per_item, reduce="sum")
    return results

def query10_agg_combine(sorted_data,groups_unique,group_idx):
    results = torch.zeros(groups_unique.shape[0],dtype=sorted_data[1].dtype,device=sorted_data[0].get_device()).scatter_reduce(0, group_idx, sorted_data[1], reduce="sum")
    return results


def run_query10_on_parquet(iteration, store, target_device, scale_factor):
    WORLD_SIZE = int(os.environ['WORLD_SIZE'])
    if os.environ["SHUFFLE_TYPE"] == "BLOCKING":
        SHUFFLE_MODULE_TYPE = ShuffleWithJoinNaive
    elif "OVERLAPPED" in os.environ["SHUFFLE_TYPE"]:
        SHUFFLE_MODULE_TYPE = ShuffleWithJoin_MultiStream
    elif os.environ["SHUFFLE_TYPE"] == "GRACE":
        # SSD spill mode (patch 0003): each join side spills to the data SSD that does not hold the table it scans
        # needed_columns: the joined columns query10_probe_pipeline reads (c_custkey, c_nationkey, c_acctbal,
        # l_extendedprice, l_discount); only these and the keys are spilled
        SHUFFLE_MODULE_TYPE = partial(GraceHashJoin, build_spill_dirs=os.environ["SPILL_DIR_OTHER_TABLES"],
                                      probe_spill_dirs=os.environ["SPILL_DIR_LINEITEM"], needed_columns=[0, 1, 2, 7, 8])
    else:
        print(f"Error, unknown SHUFFLE_TYPE={os.environ["SHUFFLE_TYPE"]}")
        return
    storage_threads = int(os.environ["STORAGE_THREADS"])

    base_folder = f"{os.environ["STORAGE_BASE_PATH"]}/tpch-sf{scale_factor}/"
    # PAGE = int(os.environ.get("PAGE")) # 10, 25, 50, 100, 200, 400, 600, 800

    reader_wrapper = ReaderWrapper(store, target_device,
                                         # name, filepath, threads, cols, filters
                                table_infos= [  
                                                ("customer", f"{base_folder}/customer.parquet", storage_threads,
                                                    ["c_custkey","c_nationkey","c_acctbal"],
                                                    None
                                                ),

                                                ("orders", f"{base_folder}/orders.parquet", storage_threads,
                                                    ["o_orderkey", "o_custkey", "o_orderdate"],
                                                    TableFilter({
                                                        2: [FilterCondition(FilterOp.GE, datetime.date(1993, 10, 1)),
                                                            FilterCondition(FilterOp.LT, datetime.date(1994, 1, 1))]  # o_orderdate
                                                    })
                                                ),

                                                # ("lineitem", f"{page_base_folder}/lineitem_10000000_pc{PAGE}.parquet", storage_threads,
                                                ("lineitem", f"{base_folder}/lineitem.parquet", storage_threads,
                                                    ["l_orderkey","l_extendedprice","l_discount","l_returnflag"],
                                                    TableFilter({
                                                        3: [FilterCondition(FilterOp.EQ, string_to_categoryidx("l_returnflag","R"))]  # l_returnflag
                                                    })
                                                )
                                        ]
                                )

    customer_reader, orders_reader, lineitem_reader = reader_wrapper.readers

    if os.environ["HASH_SIZE_NAIVE"] == "True":
        cust_order_hash_init = 160000*scale_factor
        order_lineitem_hash_init = (scale_factor * 1500000) // WORLD_SIZE
    else:
        cust_order_hash_init = util.next_prime(160000*scale_factor)
        order_lineitem_hash_init = util.next_prime((scale_factor * 1500000) // WORLD_SIZE)

    torch.cuda.empty_cache()
    import gc
    gc.collect()

    customer_orders_join = SingleNodeJoin(target_device, cust_order_hash_init, left_child=customer_reader.get_fifo(),
                                            right_child=orders_reader.get_fifo(), right_col_idx=1)
    co_lineitem_left_child = customer_orders_join
    shuffle_right_child = lineitem_reader.get_fifo()

    if os.environ.get("MATERIALIZE_BEFORE_SHUFFLE") is not None:
        print("# Adding pipeline_breaker")
        customer_orders_pb = PipelineBreaker(customer_orders_join)
        co_lineitem_left_child = customer_orders_pb
    elif (tb_size := os.environ.get("SHUFFLE_BUFFER_LEFT")) is not None:
        print("# Adding tensor buffer for left join side")
        co_lineitem_left_child = TensorBuffer(customer_orders_join, int(tb_size))
    if (tb_size := os.environ.get("SHUFFLE_BUFFER_RIGHT")) is not None:
        print("# Adding tensor buffer for right join side")
        shuffle_right_child = TensorBuffer(lineitem_reader.get_fifo(), int(tb_size))

    co_lineitem = SHUFFLE_MODULE_TYPE(target_device, order_lineitem_hash_init, co_lineitem_left_child, 3,
                                      shuffle_right_child, 0, store, store_prefix="")
    if os.environ["SHUFFLE_TYPE"] == "GRACE":
        # SSD spill mode (patch 0003): nothing probes the customer join once the Grace join has drained its build side
        co_lineitem.release_after_build_side.append(customer_orders_join)
    # joined_tuples
    # [ 0,          1,            2,          3,            4,           5,             6,           7,                8,           9]
    # ["c_custkey","c_nationkey","c_acctbal","o_orderkey", "o_custkey", "o_orderdate", "l_orderkey","l_extendedprice","l_discount","l_returnflag"]
    query10_probe_pipeline = AggregationPostCombine(query10_agg, query10_agg_combine, [0,1,2], projection=[7,8])

    ######################################################################################
    def tmp_function():
        customer_orders_join.build(customer_reader.get_fifo(), 0)
        customer_orders_join.finish()

        result = co_lineitem(probe_pipeline=query10_probe_pipeline)
        agg_finalize_start = time.perf_counter_ns()
        query10_probe_pipeline.finalize()
        torch.cuda.synchronize(device=target_device)

        result["time_finalize_agg_ns"] = time.perf_counter_ns() - agg_finalize_start
        return result

    result_str = reader_wrapper.run(tmp_function)
    print(f"# Join tuples after shuffle       : {customer_orders_join.num_join_tuples}")
    if iteration == 0:
        print(f"query,iteration,algo,{reader_wrapper.header_str}")
    print(f"query10,{iteration},{co_lineitem.name()},{result_str}")
    ######################################################################################
    if os.environ["SHUFFLE_PROBE_TYPE"] != "FULL":
        return

    # Print Aggregation Results:
    groups, aggs = query10_probe_pipeline.get_results()
    dump_result({"groups": groups, "aggs": aggs}) # FlowCSD observation, after the timed region
    torch.set_printoptions(linewidth=200,precision=4,sci_mode=False)
    print(f"# [Local] Result groups: {groups.shape=}")
    print("# c_custkey (first 5)    : ",groups[:5,0])
    # print("# o_orderdate (first 5)   : ",[util.epoch_days_to_datetime(x).strftime('%Y-%m-%d') for x in groups[:5,1].tolist()])
    print("# c_nationkey (first 5)    : ",groups[:5,1])
    print("# c_acctbal (first 5): ",groups[:5,2])
    print("# revenue (first 5)       : ",aggs[:5])

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
                print("# c_custkey (first 5)    : ",unique_groups[:5,0])
                # print("# o_orderdate (first 5)   : ",[util.epoch_days_to_datetime(x).strftime('%Y-%m-%d') for x in unique_groups[:5,1].tolist()])
                print("# c_nationkey (first 5)    : ",unique_groups[:5,1])
                print("# c_acctbal (first 5): ",unique_groups[:5,2])
                print("# revenue (first 5)       : ",final_aggs[:5])

    torch.set_printoptions()

