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
from modules.filter_module import FilterCondition, FilterOp, FilterOpCol, TableFilter
from modules.single_thread_naive import ShuffleWithJoinNaive
from modules.single_thread_shuffle_join_multistream import ShuffleWithJoin_MultiStream

from modules.query_wrapper import ReaderWrapper
from modules.join_wrapper_module import join_wrapper
from modules.aggregation_module import Aggregation,AggregationPostCombine
from modules.pipeline_breaker import PipelineBreaker
from modules.morsel_table_module import MorselTableModule
from modules.tensor_buffer import TensorBuffer
from modules.single_node_join import SingleNodeJoin
from .tpch import get_orders,get_customer,get_lineitem
from .dict import *

QUERY17_SQL = """
select
        sum(l_extendedprice) / 7.0 as avg_yearly
from
        lineitem,
        part
where
        p_partkey = l_partkey
        and p_brand = 'Brand#23'
        and p_container = 'MED BOX'
        and l_quantity < (
                select
                        0.2 * avg(l_quantity)
                from
                        lineitem
                where
                        l_partkey = p_partkey
        )
"""


def query17_agg(sorted_data,groups_unique,group_idx):
    quantity = sorted_data[1]
    ones = torch.ones(sorted_data[0].shape[0],dtype=torch.int64,device=sorted_data[0].get_device())

    results = torch.zeros((groups_unique.shape[0], 2),dtype=torch.int64,device=sorted_data[0].get_device())
    results[:,0].scatter_reduce_(0, group_idx, quantity, reduce="sum")
    results[:,1].scatter_reduce_(0, group_idx, ones, reduce="sum")
    return results

def query17_agg_combine(sorted_data,groups_unique,group_idx):
    results = torch.zeros((groups_unique.shape[0],2),dtype=sorted_data[1].dtype,device=sorted_data[0].get_device())

    results[:,0].scatter_reduce_(0, group_idx, sorted_data[1][:,0], reduce="sum")
    results[:,1].scatter_reduce_(0, group_idx, sorted_data[1][:,1], reduce="sum")
    return results

def query17_second_probe(aggregation, joined_tuples):
    # joined_tuples
    # [ 0,           1,                       2,           3,            4]
    # ["l_partkey", "0.2*(avg(l_quantity))", "l_partkey", "l_quantity", "l_extendedprice"]
    mask = torch.lt(joined_tuples[3], joined_tuples[1])

    price = torch.masked_select(joined_tuples[4],mask)

    aggregation[0] += price.sum()


def run_query17_on_parquet(iteration, store, target_device, scale_factor):
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

                                                ("part", f"{base_folder}/part.parquet", storage_threads,
                                                    ["p_partkey", "p_brand","p_container"],
                                                    TableFilter({
                                                        1: [FilterCondition(FilterOp.EQ, string_to_categoryidx("p_brand","Brand#23"))],     # p_brand = 'Brand#23'
                                                        2: [FilterCondition(FilterOp.EQ, string_to_categoryidx("p_container","MED BOX"))],  # p_container = 'MED BOX'
                                                    }),
                                                ),

                                                ("lineitem", f"{base_folder}/lineitem.parquet", storage_threads,
                                                    ["l_partkey", "l_quantity", "l_extendedprice"],
                                                    None,
                                                ),
                                                ("lineitem2", f"{base_folder}/lineitem.parquet", storage_threads,
                                                    ["l_partkey", "l_quantity", "l_extendedprice"],
                                                    None,
                                                ),
                                        ],
                                read_full_infos=[], # IMPORTANT: This means all 3 tables are read partitioned, not assumed to be replicated
                                )

    part_reader, lineitem_reader, lineitem2_reader  = reader_wrapper.readers

    if os.environ["HASH_SIZE_NAIVE"] == "True":
        lineitem_part_hash_init = (scale_factor * 4000) // WORLD_SIZE
    else:
        lineitem_part_hash_init = util.next_prime((scale_factor * 4000) // WORLD_SIZE)

    torch.cuda.empty_cache()
    import gc
    gc.collect()

    shuffle_left_child = part_reader.get_fifo()
    shuffle_right_child = lineitem_reader.get_fifo()

    if (tb_size := os.environ.get("SHUFFLE_BUFFER_LEFT")) is not None:
        print("# Adding tensor buffer for left join side")
        shuffle_left_child = TensorBuffer(part_reader.get_fifo(), int(tb_size))
    if (tb_size := os.environ.get("SHUFFLE_BUFFER_RIGHT")) is not None:
        print("# Adding tensor buffer for right join side")
        shuffle_right_child = TensorBuffer(lineitem_reader.get_fifo(), int(tb_size))

    part_lineitem_shuffle = SHUFFLE_MODULE_TYPE(target_device, lineitem_part_hash_init, shuffle_left_child, 0,
                                      shuffle_right_child, 0, store, store_prefix="")

    morsel_table = MorselTableModule([torch.empty(0,dtype=torch.int64,device=target_device),
                                      torch.empty(0,dtype=torch.int64,device=target_device)],(1<<30)) # init columns later
    # second shuffle hash init could be even smaller
    second_shuffle = SHUFFLE_MODULE_TYPE(target_device, lineitem_part_hash_init, morsel_table, 0,
                                      lineitem2_reader.get_fifo(), 0, store, store_prefix="", shuffle_left=False)

    # joined_tuples
    # [ 0,           1,        2,            3,           4,            5]
    # ["p_partkey", "p_brand","p_container","l_partkey", "l_quantity", "l_extendedprice"]
    query17_probe_pipeline = AggregationPostCombine(query17_agg, query17_agg_combine, [3], projection=[3,4])
    # joined_tuples
    # [ 0,           1,                       2,           3,            4]
    # ["l_partkey", "0.2*(avg(l_quantity))", "l_partkey", "l_quantity", "l_extendedprice"]
    query17_final_aggregation = torch.zeros(1,dtype=torch.int64,device="cuda:0")
    query17_second_probe_pipeline = partial(query17_second_probe,query17_final_aggregation)

    ######################################################################################
    def tmp_function():
        result = part_lineitem_shuffle(probe_pipeline=query17_probe_pipeline)
        query17_probe_pipeline.finalize()
        torch.cuda.synchronize(device=target_device)
        groups, aggs = query17_probe_pipeline.get_results()
        # [l_partkey,0.2 * avg(l_quantity)]
        morsel_table.columns = [groups[:,0], 0.2*torch.div(aggs[:,0],aggs[:,1])]
        torch.cuda.synchronize(device=target_device)
        result2 = second_shuffle(probe_pipeline=query17_second_probe_pipeline)

        result["time_finalize_agg_ns"] = 0
        result["time_ns"] += result2["time_ns"]
        result["time_part_ns"] =  result.get("time_part_ns",0) + result2.get("time_part_ns",-1)
        result["time_shuffle_ns"] =  result.get("time_shuffle_ns",0) + result2.get("time_shuffle_ns",-1)
        result["time_buildprobe_ns"] =  result.get("time_buildprobe_ns",0) + result2.get("time_buildprobe_ns",-1)
        result["time_finalize_agg_ns"] =  result.get("time_finalize_agg_ns",0) + result2.get("time_finalize_agg_ns", -1)
        result["shuffled_bytes"] = (result["shuffled_bytes"][0] + second_shuffle.get_shuffled_bytes()[0],
                                    result["shuffled_bytes"][1] + second_shuffle.get_shuffled_bytes()[1])
        return result

    result_str = reader_wrapper.run(tmp_function)
    print(f"# part_lineitem_shuffle (build/probe side tuples)   : {part_lineitem_shuffle.get_stats()}")
    print(f"# second_shuffle stats (build/probe side tuples)           : {second_shuffle.get_stats()}")
    if iteration == 0:
        print(f"query,iteration,algo,{reader_wrapper.header_str}")
    print(f"query17,{iteration},{part_lineitem_shuffle.name()},{result_str}")
    ######################################################################################

    if os.environ["SHUFFLE_PROBE_TYPE"] != "FULL":
        return

    ######################################################################################
    #######################################################################################
    groups, aggs = query17_probe_pipeline.get_results()
    from helpers.flowcsd_observe import dump_result
    dump_result({"sum_price": query17_final_aggregation})
    print(f"# [Local] First Result groups: {groups.shape=}")
    print("# l_partkey      :", groups[:5,0])
    print("# sum(l_quantity):", aggs[:5,0])
    print("# count(*)       :", aggs[:5,1])
    print(f"# [Local] Final Results:")
    print("# sum(l_extendedprice)  :", query17_final_aggregation[0])
    print("# sum(l_extendedprice)/7  :", query17_final_aggregation[0]/7)

    if os.environ.get("SYNC_RESULT", "False") == "True" and WORLD_SIZE > 1:
        print(f"# [Global] Results:")
        dist.all_reduce(query17_final_aggregation, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize(target_device)
        print("# sum(l_extendedprice) :", query17_final_aggregation[0])
        print("# sum(l_extendedprice)/7  :", query17_final_aggregation[0]/7)

