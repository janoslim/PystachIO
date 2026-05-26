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
from modules.join_wrapper_module import join_wrapper
from modules.aggregation_module import Aggregation,AggregationPostCombine
from modules.shuffle_feed import ShuffleFeed
from modules.pipeline_breaker import PipelineBreaker
from modules.sort_module import SortPostCombine
from modules.morsel_table_module import MorselTableModule
from modules.tensor_buffer import TensorBuffer
from modules.single_node_join import SingleNodeJoin
from .tpch import get_orders,get_customer,get_lineitem
from .dict import *

QUERY16_SQL = """
select
        p_brand,
        p_type,
        p_size,
        --count(distinct ps_suppkey) as supplier_cnt
        sum(ps_suppkey)
from
        partsupp,
        part
where
        p_partkey = ps_partkey
        and p_brand <> 'Brand#45'
        and p_type not like 'MEDIUM POLISHED%'
        and p_size in (49, 14, 23, 45, 19, 3, 36, 9)
        and ps_suppkey not in (
                select
                        s_suppkey
                from
                        supplier
                where
                        s_acctbal < -850
        )
group by
        p_brand,
        p_type,
        p_size
order by
        p_brand,
        p_type,
        p_size
"""

def query16_agg(sorted_data,groups_unique,group_idx):
    # [ 0 ]
    # ["ps_suppkey"]
    # print(sorted_data[0][:5])
    results = torch.zeros(groups_unique.shape[0],dtype=sorted_data[0].dtype,device=sorted_data[0].get_device()).scatter_reduce(0, group_idx, sorted_data[0], reduce="sum", include_self=False)
    return results

def query16_agg_combine(sorted_data,groups_unique,group_idx):
    results = torch.zeros(groups_unique.shape[0],dtype=sorted_data[1].dtype,device=sorted_data[0].get_device()).scatter_reduce(0, group_idx, sorted_data[1], reduce="sum", include_self=False)
    return results


def query16_probe_pipeline(supplier_join, aggregation, joined_tuples):
    # [ 0,           1,         2,       3,       4,           5]
    # ["p_partkey", "p_brand", "p_type","p_size","ps_partkey","ps_suppkey"]
    for final_data in supplier_join.probe(joined_tuples, 5): # join using ps_suppkey
        # ["s_suppkey","s_acctbal","p_partkey", "p_brand", "p_type","p_size","ps_partkey","ps_suppkey"]
        aggregation([final_data[3],final_data[4],final_data[5],final_data[7]])


def run_query16_on_parquet(iteration, store, target_device, scale_factor):
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

                                                ("supplier", f"{base_folder}/supplier.parquet", storage_threads,
                                                    ["s_suppkey","s_acctbal"],
                                                    TableFilter({1: [FilterCondition(FilterOp.LT, -85000)]}),
                                                ),

                                                ("part", f"{base_folder}/part.parquet", storage_threads,
                                                    ["p_partkey", "p_brand", "p_type","p_size"],
                                                    TableFilter({
                                                        1: [FilterCondition(FilterOp.NE, string_to_categoryidx("p_brand", "Brand#45"))],
                                                        2: [FilterCondition(FilterOp.NI, torch.tensor([70,71,72,73,74],device=target_device))],
                                                        3: [FilterCondition(FilterOp.II, torch.tensor([49, 14, 23, 45, 19, 3, 36, 9],device=target_device))],
                                                    })
                                                ),

                                                ("partsupp", f"{base_folder}/partsupp.parquet", storage_threads,
                                                    ["ps_partkey","ps_suppkey"],
                                                    None
                                                ),

                                        ],
                                )

    supplier_reader, part_reader, partsupp_reader = reader_wrapper.readers

    if os.environ["HASH_SIZE_NAIVE"] == "True":
        supplier_hash_init = (scale_factor * 30000) // WORLD_SIZE
        part_partsupp_hash_init = (scale_factor * 40000) // WORLD_SIZE
    else:
        supplier_hash_init = util.next_prime((scale_factor * 30000) // WORLD_SIZE)
        part_partsupp_hash_init = util.next_prime((scale_factor * 40000) // WORLD_SIZE)

    torch.cuda.empty_cache()
    import gc
    gc.collect()

    # ["n_nationkey", "n_regionkey","s_suppkey","s_nationkey", REST]
    supplier_join = SingleNodeJoin(target_device, supplier_hash_init, left_child=supplier_reader.get_fifo(),
                                    anti_join=True)

    shuffle_left_child = part_reader.get_fifo()
    shuffle_right_child = partsupp_reader.get_fifo()

    if (tb_size := os.environ.get("SHUFFLE_BUFFER_LEFT")) is not None:
        print("# Adding tensor buffer for left join side")
        shuffle_left_child = TensorBuffer(part_reader.get_fifo(), int(tb_size))
    if (tb_size := os.environ.get("SHUFFLE_BUFFER_RIGHT")) is not None:
        print("# Adding tensor buffer for right join side")
        shuffle_right_child = TensorBuffer(partsupp_reader.get_fifo(), int(tb_size))

    shufflejoin = SHUFFLE_MODULE_TYPE(target_device, part_partsupp_hash_init, shuffle_left_child, 0,
                                                    shuffle_right_child, 0, store, store_prefix="")
    # joined_tuples
    # [ 0,          1,        2,           3,           4,            ]
    # ["p_partey", "p_size", "ps_partkey","ps_suppkey","ps_supplycost"]
    query16_aggregation = AggregationPostCombine(query16_agg, query16_agg_combine, [0,1,2], projection=[3])
    probe_pipeline = partial(query16_probe_pipeline, supplier_join, query16_aggregation)

    ######################################################################################
    def tmp_function():
        supplier_join.build(supplier_reader.get_fifo(), 0)
        supplier_join.finish()

        result = shufflejoin(probe_pipeline=probe_pipeline)
        agg_finalize_start = time.perf_counter_ns()
        query16_aggregation.finalize()
        torch.cuda.synchronize(device=target_device)

        result["time_finalize_agg_ns"] = time.perf_counter_ns() - agg_finalize_start
        return result

    result_str = reader_wrapper.run(tmp_function)
    print(f"# shufflejoin tuples        : {shufflejoin.get_stats()}")
    print(f"# supplier_join             : {supplier_join.get_stats()}")
    if iteration == 0:
        print(f"query,iteration,algo,{reader_wrapper.header_str}")
    print(f"query16,{iteration},{shufflejoin.name()},{result_str}")
    ######################################################################################
    if os.environ["SHUFFLE_PROBE_TYPE"] != "FULL":
        return

    # Print Aggregation Results:
    groups, aggs = query16_aggregation.get_results()
    torch.set_printoptions(linewidth=200,precision=4,sci_mode=False)
    print(f"# [Local] Result groups: {groups.shape=}")
    print("# p_brand (first 5)          : ",groups[:5,0])
    print("# p_type (first 5)           : ",groups[:5,1])
    print("# p_size (first 5)           : ",groups[:5,2])
    print("# min(ps_supplycost) (first 5): ",aggs[:5])

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
    #             print("# l_orderkey (first 5)    : ",unique_groups[:5,0])
    #             # print("# o_orderdate (first 5)   : ",[util.epoch_days_to_datetime(x).strftime('%Y-%m-%d') for x in unique_groups[:5,1].tolist()])
    #             print("# o_shippriority (first 5): ",unique_groups[:5,2])
    #             print("# revenue (first 5)       : ",final_aggs[:5])

    torch.set_printoptions()

