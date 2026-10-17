import sys
import os
import time
import datetime
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
from functools import partial
import pandas as pd
import torch
import torch.distributed as dist

from helpers import util
from helpers.flowcsd_observe import dump_result
from modules.filter_module import FilterCondition, FilterOp, FilterOpCol, TableFilter
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

QUERY14_SQL = """
select
        100.00 * sum(case
                when p_type like 'PROMO%'
                        then l_extendedprice * (1 - l_discount)
                else 0
        end) / sum(l_extendedprice * (1 - l_discount)) as promo_revenue
from
        lineitem,
        part
where
        l_partkey = p_partkey
        and l_shipdate >= date '1995-09-01'
        and l_shipdate < date '1995-09-01' + interval '1' month
"""


def query14_probe_pipeline(aggregation, joined_tuples):
     # joined_tuples
    # [ 0,           1,                2,            3,           4,           5]
    # ["l_partkey", "l_extendedprice","l_discount", "l_shipdate","p_partkey", "p_type"]
    # sum(l_extendedprice * (1 - l_discount)) as revenue

    discount_factor = (100 - joined_tuples[2])
    revenue_per_item = joined_tuples[1] * discount_factor
    # joined_tuples[5] like "PROMO%" -> [75-99]
    promos = (torch.ge(joined_tuples[5],75) & torch.le(joined_tuples[5], 99)).to(torch.int8)
    revenue_per_promo = revenue_per_item * promos

    aggregation[0] += revenue_per_promo.sum()
    aggregation[1] += revenue_per_item.sum()



def run_query14_on_parquet(iteration, store, target_device, scale_factor):
    WORLD_SIZE = int(os.environ['WORLD_SIZE'])
    if os.environ["SHUFFLE_TYPE"] == "BLOCKING":
        SHUFFLE_MODULE_TYPE = ShuffleWithJoinNaive
    elif "OVERLAPPED" in os.environ["SHUFFLE_TYPE"]:
        SHUFFLE_MODULE_TYPE = ShuffleWithJoin_MultiStream
    elif os.environ["SHUFFLE_TYPE"] == "GRACE":
        # SSD spill mode (patch 0003): each join side spills to the data SSD that does not hold the table it scans
        # needed_columns: the joined columns query14_probe_pipeline reads (l_extendedprice, l_discount, p_type); only
        # these and the keys are spilled
        SHUFFLE_MODULE_TYPE = partial(GraceHashJoin, build_spill_dirs=os.environ["SPILL_DIR_LINEITEM"],
                                      probe_spill_dirs=os.environ["SPILL_DIR_OTHER_TABLES"], needed_columns=[1, 2, 5])
    else:
        print(f"Error, unknown SHUFFLE_TYPE={os.environ["SHUFFLE_TYPE"]}")
        return
    storage_threads = int(os.environ["STORAGE_THREADS"])

    base_folder = f"{os.environ["STORAGE_BASE_PATH"]}/tpch-sf{scale_factor}/"

    reader_wrapper = ReaderWrapper(store, target_device,
                                         # name, filepath, threads, cols, filters
                                table_infos= [  

                                                ("part", f"{base_folder}/part.parquet", storage_threads,
                                                    ["p_partkey", "p_type"],
                                                    None,
                                                ),

                                                ("lineitem", f"{base_folder}/lineitem.parquet", storage_threads,
                                                    ["l_partkey", "l_extendedprice","l_discount", "l_shipdate"],
                                                    TableFilter({
                                                        3: [FilterCondition(FilterOp.GE, datetime.date(1995, 9, 1)),    # l_shipdate >= date '1995-09-01'
                                                            FilterCondition(FilterOp.LT, datetime.date(1995, 10, 1))],  # l_shipdate < date '1995-10-01'
                                                    })
                                                ),
                                        ]
                                )

    part_reader, lineitem_reader = reader_wrapper.readers

    if os.environ["HASH_SIZE_NAIVE"] == "True":
        lineitem_part_hash_init = (scale_factor * 1000000) // WORLD_SIZE
    else:
        lineitem_part_hash_init = util.next_prime((scale_factor * 1000000) // WORLD_SIZE)

    torch.cuda.empty_cache()
    import gc
    gc.collect()

    shuffle_left_child = lineitem_reader.get_fifo()
    shuffle_right_child = part_reader.get_fifo()

    if (tb_size := os.environ.get("SHUFFLE_BUFFER_LEFT")) is not None:
        print("# Adding tensor buffer for left join side")
        shuffle_left_child = TensorBuffer(lineitem_reader.get_fifo(), int(tb_size))
    if (tb_size := os.environ.get("SHUFFLE_BUFFER_RIGHT")) is not None:
        print("# Adding tensor buffer for right join side")
        shuffle_right_child = TensorBuffer(part_reader.get_fifo(), int(tb_size))

    linitem_part_shuffle = SHUFFLE_MODULE_TYPE(target_device, lineitem_part_hash_init, shuffle_left_child, 0,
                                      shuffle_right_child, 0, store, store_prefix="")
    # joined_tuples
    # [ 0,           1,                2,            3,           4,           5]
    # ["l_partkey", "l_extendedprice","l_discount", "l_shipdate","p_partkey", "p_type"]
    query14_aggregation = torch.zeros(2,dtype=torch.int64,device="cuda:0")
    probe_pipeline = partial(query14_probe_pipeline, query14_aggregation)

    ######################################################################################
    def tmp_function():
        return linitem_part_shuffle(probe_pipeline=probe_pipeline)

    result_str = reader_wrapper.run(tmp_function)
    if iteration == 0:
        print(f"query,iteration,algo,{reader_wrapper.header_str}")
    print(f"query14,{iteration},{linitem_part_shuffle.name()},{result_str}")
    ######################################################################################

    if os.environ["SHUFFLE_PROBE_TYPE"] != "FULL":
        return

    ######################################################################################
    dump_result({"aggregation": query14_aggregation}) # FlowCSD observation, after the timed region
    print(f"# [Local] Results:")
    print("# promos revenue :", query14_aggregation[0])
    print("# total revenue  :", query14_aggregation[1])
    print("# percent        :", 100.0 * query14_aggregation[0].item()/query14_aggregation[1].item())

    if os.environ.get("SYNC_RESULT", "False") == "True" and WORLD_SIZE > 1:
        print(f"# [Global] Results:")
        dist.all_reduce(query14_aggregation, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize(target_device)
        print("# promos revenue :", query14_aggregation[0])
        print("# total revenue  :", query14_aggregation[1])
        print("# percent        :", 100.0 * query14_aggregation[0].item()/query14_aggregation[1].item())

