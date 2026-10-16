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

from modules.query_wrapper import ReaderWrapper
from modules.join_wrapper_module import join_wrapper
from modules.aggregation_module import Aggregation,AggregationPostCombine
from modules.pipeline_breaker import PipelineBreaker
from modules.tensor_buffer import TensorBuffer
from modules.single_node_join import SingleNodeJoin
from .tpch import get_orders,get_customer,get_lineitem
from .dict import *

QUERY19_SQL = """
select
        sum(l_extendedprice* (1 - l_discount)) as revenue
from
        lineitem,
        part
where
        (
                p_partkey = l_partkey
                and p_brand = 'Brand#12'
                and p_container in ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG')
                and l_quantity >= 1 and l_quantity <= 1 + 10
                and p_size between 1 and 5
                and l_shipmode in ('AIR', 'AIR REG')
                and l_shipinstruct = 'DELIVER IN PERSON'
        )
        or
        (
                p_partkey = l_partkey
                and p_brand = 'Brand#23'
                and p_container in ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK')
                and l_quantity >= 10 and l_quantity <= 10 + 10
                and p_size between 1 and 10
                and l_shipmode in ('AIR', 'AIR REG')
                and l_shipinstruct = 'DELIVER IN PERSON'
        )
        or
        (
                p_partkey = l_partkey
                and p_brand = 'Brand#34'
                and p_container in ('LG CASE', 'LG BOX', 'LG PACK', 'LG PKG')
                and l_quantity >= 20 and l_quantity <= 20 + 10
                and p_size between 1 and 15
                and l_shipmode in ('AIR', 'AIR REG')
                and l_shipinstruct = 'DELIVER IN PERSON'
        )
"""


def query19_probe_pipeline(aggregation, joined_tuples):
    # joined_tuples
    # [ 0,           1,            2,                3,            4,                5,            6,           7,         8,        9]
    # ["l_partkey", "l_quantity", "l_extendedprice","l_discount", "l_shipinstruct", "l_shipmode", "p_partkey", "p_brand", "p_size", "p_container"]
    # sum(l_extendedprice * (1 - l_discount)) as revenue

    discount_factor = (100 - joined_tuples[3])
    revenue = joined_tuples[2] * discount_factor

    sm_case_vals = torch.tensor([     
                                 string_to_categoryidx("p_container","SM CASE"),
                                 string_to_categoryidx("p_container","SM BOX"),
                                 string_to_categoryidx("p_container","SM PACK"),
                                 string_to_categoryidx("p_container","SM PKG")
                                ],device=joined_tuples[0].get_device())

    med_case_vals = torch.tensor([
                                 string_to_categoryidx("p_container","MED BAG"),
                                 string_to_categoryidx("p_container","MED BOX"),
                                 string_to_categoryidx("p_container","MED PKG"),
                                 string_to_categoryidx("p_container","MED PACK")
                                 ],device=joined_tuples[0].get_device())
    lg_case_vals = torch.tensor([
                                 string_to_categoryidx("p_container","LG CASE"),
                                 string_to_categoryidx("p_container","LG BOX"),
                                 string_to_categoryidx("p_container","LG PACK"),
                                 string_to_categoryidx("p_container","LG PKG")
                                 ],device=joined_tuples[0].get_device())

    part_1 = (torch.eq(joined_tuples[7],string_to_categoryidx("p_brand","Brand#12"))
            & torch.isin(joined_tuples[9], sm_case_vals)
            & torch.le(joined_tuples[8], 5)
            & torch.le(joined_tuples[1], 1100))

    part_2 = (torch.eq(joined_tuples[7],string_to_categoryidx("p_brand","Brand#23"))
            & torch.isin(joined_tuples[9], med_case_vals)
            & torch.le(joined_tuples[8], 10)
            & torch.ge(joined_tuples[1], 1000)
            & torch.le(joined_tuples[1], 2000))

    part_3 = (torch.eq(joined_tuples[7],string_to_categoryidx("p_brand","Brand#34")) 
            & torch.isin(joined_tuples[9], lg_case_vals)
            & torch.le(joined_tuples[8], 15)
            & torch.ge(joined_tuples[1], 2000)
            & torch.le(joined_tuples[1], 3000))

    aggregation[0] += torch.masked_select(revenue,(part_1|part_2|part_3)).sum()



def run_query19_on_parquet(iteration, store, target_device, scale_factor):
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

    reader_wrapper = ReaderWrapper(store, target_device,
                                         # name, filepath, threads, cols, filters
                                table_infos= [  
                                                ("lineitem", f"{base_folder}/lineitem.parquet", storage_threads,
                                                    ["l_partkey", "l_quantity", "l_extendedprice","l_discount", "l_shipinstruct", "l_shipmode"],
                                                    TableFilter({
                                                        1: [FilterCondition(FilterOp.LE, 3000)],  # l_quantity <= 30
                                                        4: [FilterCondition(FilterOp.EQ, string_to_categoryidx("l_shipinstruct","DELIVER IN PERSON"))],  # l_shipinstruct = 'DELIVER IN PERSON'
                                                        5: [FilterCondition(FilterOp.EQ, string_to_categoryidx("l_shipmode","AIR"))],  # l_shipmode = ('AIR', 'AIR REG')
                                                        # "The TPC recognizes that the predicates on l_shipmode include the non-existing shipmode “AIR REG”."

                                                    })
                                                ),

                                                ("part", f"{base_folder}/part.parquet", storage_threads,
                                                    ["p_partkey", "p_brand", "p_size", "p_container"],
                                                    TableFilter({
                                                        2: [FilterCondition(FilterOp.LE, 15)],  # p_size <= 15
                                                    })
                                                ),

                                        ]
                                )

    lineitem_reader, part_reader  = reader_wrapper.readers

    if os.environ["HASH_SIZE_NAIVE"] == "True":
        lineitem_part_hash_init = (scale_factor * 160000) // WORLD_SIZE
    else:
        lineitem_part_hash_init = util.next_prime((scale_factor * 160000) // WORLD_SIZE)

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
    # [ 0,           1,            2,                3,            4,                5,            6,           7,         8,        9]
    # ["l_partkey", "l_quantity", "l_extendedprice","l_discount", "l_shipinstruct", "l_shipmode", "p_partkey", "p_brand", "p_size", "p_container"]
    query19_aggregation = torch.zeros(1,dtype=torch.int64,device="cuda:0")
    probe_pipeline = partial(query19_probe_pipeline, query19_aggregation)

    ######################################################################################
    def tmp_function():
        return linitem_part_shuffle(probe_pipeline=probe_pipeline)

    result_str = reader_wrapper.run(tmp_function)
    if iteration == 0:
        print(f"query,iteration,algo,{reader_wrapper.header_str}")
    print(f"query19,{iteration},{linitem_part_shuffle.name()},{result_str}")
    ######################################################################################

    if os.environ["SHUFFLE_PROBE_TYPE"] != "FULL":
        return

    ######################################################################################
    dump_result({"aggregation": query19_aggregation}) # FlowCSD observation, after the timed region
    print(f"# [Local] Results:")
    print("# revenue :", query19_aggregation[0])

    if os.environ.get("SYNC_RESULT", "False") == "True" and WORLD_SIZE > 1:
        print(f"# [Global] Results:")
        dist.all_reduce(query19_aggregation, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize(target_device)
        print("# revenue :", query19_aggregation[0])

