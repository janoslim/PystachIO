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

from modules.query_wrapper import ReaderWrapper
from modules.feed import Feed
from modules.join_wrapper_module import join_wrapper
from modules.aggregation_module import Aggregation,AggregationPostCombine
from modules.pipeline_breaker import PipelineBreaker
from modules.single_node_join import SingleNodeJoin
from .tpch import get_orders,get_customer,get_lineitem
from .dict import *

# almost query 1
QUERY6_SQL = """
select
        sum(l_extendedprice * l_discount) as revenue
from
        lineitem
where
        l_shipdate >= date '1994-01-01'
        and l_shipdate < date '1994-01-01' + interval '1' year
        and l_discount between 0.06 - 0.01 and 0.06 + 0.01
        and l_quantity < 24
"""

def query6_probe_pipeline(aggregation, joined_tuples):
    # joined_tuples
    # [ 0,           1,                2,           3]
    # ["l_quantity","l_extendedprice","l_discount","l_shipdate"]
    price = joined_tuples[1]
    revenue_per_item = price * joined_tuples[2]

    aggregation[0] += revenue_per_item.sum()


def run_query6_on_parquet(iteration, store, target_device, scale_factor):
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
                                                # ("lineitem", f"{page_base_folder}/lineitem_10000000_pc{PAGE}.parquet", storage_threads,
                                                ("lineitem", f"{base_folder}/lineitem.parquet", storage_threads,
                                                ["l_quantity","l_extendedprice","l_discount","l_shipdate"],
                                                TableFilter({
                                                    0: [FilterCondition(FilterOp.LT, 2400)], # l_quantity
                                                    2: [FilterCondition(FilterOp.GE, 5),
                                                        FilterCondition(FilterOp.LE, 7)], # l_discount
                                                    3: [FilterCondition(FilterOp.GE, datetime.date(1994,1,1)),
                                                        FilterCondition(FilterOp.LT, datetime.date(1995,1,1))]  # l_shipdate
                                                }))
                                        ]
                                )

    lineitem_reader = reader_wrapper.readers[0]

    torch.cuda.empty_cache()
    import gc
    gc.collect()

    root_operator = Feed(lineitem_reader.get_fifo(), target_device)
    query6_aggregation = torch.zeros(1,dtype=torch.float64,device="cuda:0")
    probe_pipeline = partial(query6_probe_pipeline, query6_aggregation)

    ######################################################################################
    def tmp_function():

        return root_operator(probe_pipeline=probe_pipeline)

    result_str = reader_wrapper.run(tmp_function)
    print(f"# Tuples local: {root_operator.seen_tuples}")
    if iteration == 0:
        print(f"query,iteration,algo,{reader_wrapper.header_str}")
    print(f"query6,{iteration},{root_operator.name()},{result_str}")
    ######################################################################################
    if os.environ["SHUFFLE_PROBE_TYPE"] != "FULL":
        return
    ######################################################################################
    dump_result({"revenue": query6_aggregation}) # FlowCSD observation, after the timed region
    print(f"# [Local] Result Revenue: {query6_aggregation[0]}")
    if os.environ.get("SYNC_RESULT", "False") == "True" and WORLD_SIZE > 1:
        all_revenues = [None] * WORLD_SIZE
        dist.all_gather_object(all_revenues, query6_aggregation[0].item())
        if dist.get_rank() == 0: # global results in node0
            total_revenue = sum(all_revenues)
            print(f"# [Global] Result Revenue: {total_revenue}")



