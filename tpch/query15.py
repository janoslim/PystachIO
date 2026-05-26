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

from modules.query_wrapper import ReaderWrapper
from modules.feed import Feed
from modules.shuffle_feed import ShuffleFeed
from modules.join_wrapper_module import join_wrapper
from modules.aggregation_module import Aggregation,AggregationPostCombine
from modules.pipeline_breaker import PipelineBreaker
from modules.morsel_table_module import MorselTableModule
from modules.single_node_join import SingleNodeJoin
from .tpch import get_orders,get_customer,get_lineitem
from .dict import *

# almost query 1
QUERY15_SQL = """
with revenue (supplier_no, total_revenue) as (
        select
                l_suppkey,
                sum(l_extendedprice * (1 - l_discount))
        from
                lineitem
        where
                l_shipdate >= date '1996-01-01'
                and l_shipdate < date '1996-01-01' + interval '3' month
        group by
                l_suppkey)
select
        s_suppkey,
        s_acctbal,
        total_revenue
from
        supplier,
        revenue
where
        s_suppkey = supplier_no
        and total_revenue = (
                select
                        max(total_revenue)
                from
                        revenue
        )
"""

def query15_agg(sorted_data,groups_unique,group_idx):
    # sum(l_extendedprice * (1 - l_discount)) as revenue
    revenue_per_item = sorted_data[0] * (100 - sorted_data[1])
    results = torch.zeros(groups_unique.shape[0],dtype=revenue_per_item.dtype,device=sorted_data[0].get_device()).scatter_reduce(0, group_idx, revenue_per_item, reduce="sum")
    return results

def query15_agg_combine(sorted_data,groups_unique,group_idx):
    results = torch.zeros(groups_unique.shape[0],dtype=sorted_data[1].dtype,device=sorted_data[0].get_device()).scatter_reduce(0, group_idx, sorted_data[1], reduce="sum")
    return results


def query15_last_pipeline(supplier_join, tuples):
    for final_data in supplier_join.probe(tuples, 0):
        print(f"# Result supplier: {final_data}")


def run_query15_on_parquet(iteration, store, target_device, scale_factor):
    WORLD_SIZE = int(os.environ['WORLD_SIZE'])
    if os.environ["SHUFFLE_TYPE"] == "BLOCKING":
        SHUFFLE_MODULE_TYPE = ShuffleWithJoinNaive
    elif "OVERLAPPED" in os.environ["SHUFFLE_TYPE"]:
        SHUFFLE_MODULE_TYPE = ShuffleWithJoin_MultiStream
    else:
        print(f"Error, unknown SHUFFLE_TYPE={os.environ["SHUFFLE_TYPE"]}")
        return
    storage_threads = int(os.environ["STORAGE_THREADS"])

    base_folder = f"{os.environ["STORAGE_BASE_PATH"]}/tpch-sf{scale_factor}/"
    # PAGE = int(os.environ.get("PAGE")) # 10, 25, 50, 100, 200, 400, 600, 800

    reader_wrapper = ReaderWrapper(store, target_device,
                                         # name, filepath, threads, cols, filters
                                table_infos= [

                                                ("lineitem", f"{base_folder}/lineitem.parquet", storage_threads,
                                                    ["l_suppkey","l_extendedprice","l_discount","l_shipdate"],
                                                    TableFilter({
                                                        3: [FilterCondition(FilterOp.GE, datetime.date(1996,1,1)),
                                                            FilterCondition(FilterOp.LT, datetime.date(1996,4,1))]  # l_shipdate
                                                    })
                                                ),

                                                ("supplier", f"{base_folder}/supplier.parquet", storage_threads,
                                                    # ["s_suppkey","s_name","s_address","s_phone"], None # cant read the string attributes, read the others instead
                                                    ["s_suppkey","s_nationkey","s_acctbal"], None
                                                )
                                        ],
                                read_full_infos = ["supplier"]
                                )

    lineitem_reader, supplier_reader = reader_wrapper.readers

    if os.environ["HASH_SIZE_NAIVE"] == "True":
        supplier_hash_init = (scale_factor * 12000) // WORLD_SIZE
    else:
        supplier_hash_init = util.next_prime((scale_factor * 12000) // WORLD_SIZE)

    torch.cuda.empty_cache()
    import gc
    gc.collect()

    root_operator = ShuffleFeed(target_device, lineitem_reader.get_fifo(), 0)

    dummy_table = MorselTableModule([torch.empty(0,dtype=torch.int64),torch.empty(0,dtype=torch.int64)], -1)
    supplier_join = SingleNodeJoin(target_device, supplier_hash_init, left_child=dummy_table)
    supplier_feed = Feed(supplier_reader.get_fifo(), target_device)

    # query15_aggregation = torch.zeros(1,dtype=torch.float64,device="cuda:0")
    # probe_pipeline = partial(query15_probe_pipeline, query15_aggregation)
    # ["l_suppkey","l_extendedprice","l_discount","l_shipdate"]
    query15_probe_pipeline = AggregationPostCombine(query15_agg, query15_agg_combine, [0], projection=[1,2])
    query15_final_pipeline = partial(query15_last_pipeline, supplier_join) 

    ######################################################################################
    def tmp_function():
        result = root_operator(probe_pipeline=query15_probe_pipeline)

        agg_finalize_start = time.perf_counter_ns()
        query15_probe_pipeline.finalize()

        # max(revenue)
        suppkeys, aggs = query15_probe_pipeline.get_results()

        indices = torch.argmax(aggs)
        # max_local_revenue = torch.tensor([aggs.max()], dtype=torch.int64, device=target_device)
        max_local_revenue = torch.reshape(aggs[indices], (-1,))
        best_suppkeys = torch.reshape(suppkeys[indices,0],(-1,))
        # dist.all_reduce(max_local_revenue, op=dist.ReduceOp.MAX)
        torch.cuda.synchronize(device=target_device)
        # max_revenue = max_local_revenue.cpu().item()
        # print(f"{max_local_revenue=}, {best_suppkeys=}")
        result["time_finalize_agg_ns"] = time.perf_counter_ns() - agg_finalize_start
        # do all_gather here

        # find the supplier
        supplier_join.build(MorselTableModule([best_suppkeys,max_local_revenue], (1<<31)), 0)
        supplier_join.finish()

        result = supplier_feed(probe_pipeline=query15_final_pipeline)

        return result

    result_str = reader_wrapper.run(tmp_function)
    print(f"# Tuples local (after shuffle): {root_operator.num_tuples.sum().cpu().item()}")
    if iteration == 0:
        print(f"query,iteration,algo,{reader_wrapper.header_str}")
    print(f"query15,{iteration},{root_operator.name()},{result_str}")
    ######################################################################################
    if os.environ["SHUFFLE_PROBE_TYPE"] != "FULL":
        return
    ######################################################################################
    groups, aggs = query15_probe_pipeline.get_results()
    torch.set_printoptions(linewidth=200,precision=4,sci_mode=False)
    print(f"# [Local] Result groups: {groups.shape=}")
    print("# l_suppkey (first 5)    : ",groups[:5])
    print("# revenue (first 5)      : ",aggs[:5])

    # if os.environ.get("SYNC_RESULT", "False") == "True" and WORLD_SIZE > 1:
    #     all_revenues = [None] * WORLD_SIZE
    #     dist.all_gather_object(all_revenues, query15_aggregation[0].item())
    #     if dist.get_rank() == 0: # global results in node0
    #         total_revenue = sum(all_revenues)
    #         print(f"# [Global] Result Revenue: {total_revenue}")



