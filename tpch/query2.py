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

# almost query 2
QUERY2_SQL = """

with min_suppcost_per_part (pkey, min_suppcost) as (
    select
            p_partkey,
            min(ps_supplycost)
    from
            part,
            partsupp,
            supplier,
            nation,
            region
    where
            p_partkey = ps_partkey
            and p_size = 15
            and s_suppkey = ps_suppkey
            and s_nationkey = n_nationkey
            and n_regionkey = r_regionkey
            and r_name = 'EUROPE'
    group by p_partkey
    order by p_partkey
)
select 
        s_acctbal, s_suppkey, p_partkey
from
        part,
        partsupp,
        min_suppcost_per_part,
        supplier,
        nation,
        region
where
        p_partkey = min_suppcost_per_part.pkey
        and p_partkey = ps_partkey
        and ps_supplycost = min_suppcost_per_part.min_suppcost
        and s_suppkey = ps_suppkey
        and s_nationkey = n_nationkey
        and n_regionkey = r_regionkey
        and r_name = 'EUROPE'
order by
        s_acctbal,
        s_suppkey,
        p_partkey
"""

def query2_agg(sorted_data,groups_unique,group_idx):
    # [ 0 ]
    # ["ps_supplycost"]
    # print(sorted_data[0][:5])
    results = torch.zeros(groups_unique.shape[0],dtype=sorted_data[0].dtype,device=sorted_data[0].get_device()).scatter_reduce(0, group_idx, sorted_data[0], reduce="amin", include_self=False)
    return results

def query2_agg_combine(sorted_data,groups_unique,group_idx):
    results = torch.zeros(groups_unique.shape[0],dtype=sorted_data[1].dtype,device=sorted_data[0].get_device()).scatter_reduce(0, group_idx, sorted_data[1], reduce="amin", include_self=False)
    return results


def query2_probe_pipeline(last_join, aggregation, joined_tuples):
    # after customer join:
    # [ 0,           1,        2,           3,           4,            ]
    # ["p_partkey", "p_size", "ps_partkey","ps_suppkey","ps_supplycost"]
    # after nation, supplier, prepends: ["n_nationkey", "n_regionkey","s_suppkey","s_nationkey","s_acctbal"]
    for final_data in last_join.probe(joined_tuples, 3): # join (NxS) using ps_suppkey
        # ["n_nationkey", "n_regionkey","s_suppkey","s_nationkey","s_acctbal","p_partkey", "p_size", "ps_partkey","ps_suppkey","ps_supplycost"]
        aggregation([final_data[5],final_data[9]])

def query2_probe_pipeline2(min_cost_join, last_join, sort_op, joined_tuples):
    # [ 0,           1,           2]
    # ["ps_partkey","ps_suppkey","ps_supplycost"]

    for final_data in min_cost_join.probe(joined_tuples, 0):
        for final_data2 in last_join.probe(final_data, 3):
            # [ 0,             1,            2,          3,            4,          5,          6,                 7,          8 
            # ["n_nationkey", "n_regionkey","s_suppkey","s_nationkey","s_acctbal","p_partkey","minps_supplycost","ps_partkey","ps_suppkey","ps_supplycost"]
            sort_op([final_data2[4],final_data2[2],final_data2[5]])


def run_query2_on_parquet(iteration, store, target_device, scale_factor):
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
                                                ("nation", f"{base_folder}/nation.parquet", storage_threads,
                                                    ["n_nationkey", "n_regionkey"],
                                                    TableFilter({
                                                        # r_name = r_regionkey = n_regionkey = EUROPE
                                                        1: [FilterCondition(FilterOp.EQ, string_to_categoryidx("r_name","EUROPE"))]
                                                    })
                                                ),

                                                ("supplier", f"{base_folder}/supplier.parquet", storage_threads,
                                                    ["s_suppkey","s_nationkey", "s_acctbal"],
                                                    None
                                                ),

                                                ("part", f"{base_folder}/part.parquet", storage_threads,
                                                    ["p_partkey", "p_size"],
                                                    TableFilter({
                                                        1: [FilterCondition(FilterOp.EQ, 15)]
                                                    })
                                                ),

                                                ("partsupp", f"{base_folder}/partsupp.parquet", storage_threads,
                                                    ["ps_partkey","ps_suppkey","ps_supplycost"],
                                                    None
                                                ),

                                                ("partsupp2", f"{base_folder}/partsupp.parquet", storage_threads,
                                                    ["ps_partkey","ps_suppkey","ps_supplycost"],
                                                    None
                                                )
                                        ],
                                read_full_infos = ["nation","supplier"]
                                )

    nation_reader, supplier_reader, part_reader, partsupp_reader, partsupp_reader2 = reader_wrapper.readers

    if os.environ["HASH_SIZE_NAIVE"] == "True":
        nation_supplier_hash_init = 50*scale_factor
        last_hash_init = (scale_factor * 3000) // WORLD_SIZE
        part_partsupp_hash_init = (scale_factor * 10000) // WORLD_SIZE
        after_agg_hash_init = (scale_factor * 3000) // WORLD_SIZE
    else:
        nation_supplier_hash_init = util.next_prime(50*scale_factor)
        last_hash_init = util.next_prime((scale_factor * 3000) // WORLD_SIZE)
        part_partsupp_hash_init = util.next_prime((scale_factor * 10000) // WORLD_SIZE)
        after_agg_hash_init = util.next_prime((scale_factor * 3000) // WORLD_SIZE)

    torch.cuda.empty_cache()
    import gc
    gc.collect()

    # ["n_nationkey", "n_regionkey","s_suppkey","s_nationkey"]
    nation_supplier_join = SingleNodeJoin(target_device, nation_supplier_hash_init, left_child=nation_reader.get_fifo(),
                                            right_child=supplier_reader.get_fifo(), right_col_idx=1)

    # ["n_nationkey", "n_regionkey","s_suppkey","s_nationkey", REST]
    last_join = SingleNodeJoin(target_device, last_hash_init, left_child=nation_supplier_join)

    shuffle_left_child = part_reader.get_fifo()
    shuffle_right_child = partsupp_reader.get_fifo()
    # shuffle2_left_child = part_reader2.get_fifo()
    shuffle2_right_child = partsupp_reader2.get_fifo()

    if (tb_size := os.environ.get("SHUFFLE_BUFFER_LEFT")) is not None:
        print("# Adding tensor buffer for left join side")
        shuffle_left_child = TensorBuffer(part_reader.get_fifo(), int(tb_size))
        # shuffle2_left_child = TensorBuffer(part_reader2.get_fifo(), int(tb_size))
    if (tb_size := os.environ.get("SHUFFLE_BUFFER_RIGHT")) is not None:
        print("# Adding tensor buffer for right join side")
        shuffle_right_child = TensorBuffer(partsupp_reader.get_fifo(), int(tb_size))
        shuffle2_right_child = TensorBuffer(partsupp_reader2.get_fifo(), int(tb_size))

    part_partsupp_shufflejoin = SHUFFLE_MODULE_TYPE(target_device, part_partsupp_hash_init, shuffle_left_child, 0,
                                                    shuffle_right_child, 0, store, store_prefix="")
    # joined_tuples
    # [ 0,          1,        2,           3,           4,            ]
    # ["p_partey", "p_size", "ps_partkey","ps_suppkey","ps_supplycost"]
    query2_aggregation = AggregationPostCombine(query2_agg, query2_agg_combine, [0], projection=[1])
    probe_pipeline = partial(query2_probe_pipeline, last_join, query2_aggregation)

    # AFTER GROUPBY OPERATORS
    # []
    # ["p_partkey", "p_size", "ps_partkey","ps_suppkey","ps_supplycost"]
    partsupp_again = ShuffleFeed(target_device, shuffle2_right_child, 0)

    dummy_table = MorselTableModule([torch.empty(0,dtype=torch.int64), torch.empty(0,dtype=torch.int64)], -1)
    min_cost_join = SingleNodeJoin(target_device, after_agg_hash_init, left_child=dummy_table,
                                    add_join_attrib=[(1,2)]) # join min(supplycost) == ps_supplycost
    sort_op = SortPostCombine(project=None,sort_col_idxs=[0])
    probe_pipeline2 = partial(query2_probe_pipeline2, min_cost_join, last_join, sort_op)

    ######################################################################################
    def tmp_function():
        nation_supplier_join.build(nation_reader.get_fifo(), 0)
        nation_supplier_join.finish()
        last_join.build(nation_supplier_join, 2)
        last_join.finish()

        result = part_partsupp_shufflejoin(probe_pipeline=probe_pipeline)
        agg_finalize_start = time.perf_counter_ns()
        query2_aggregation.finalize()
        torch.cuda.synchronize(device=target_device)

        result["time_finalize_agg_ns"] = time.perf_counter_ns() - agg_finalize_start
        ## subquery done, do final
        print(f"# last_join                        : {last_join.num_join_tuples}")
        last_join.num_join_tuples = 0 # reset, join gets reused

        # todo use last_join build side cols to join again (or even just use last_join?)
        groups, aggs = query2_aggregation.get_results() # pkey,min(supplycost)
        # create a join to filter tuples after second shuffle:
        min_cost_join.build(MorselTableModule([groups[:,0],aggs], (1<<31)), 0)
        min_cost_join.finish()
        result2 = partsupp_again(probe_pipeline=probe_pipeline2)
        sort_op.finalize()

        return result

    result_str = reader_wrapper.run(tmp_function)
    print(f"# nation_supplier_join tuples      : {nation_supplier_join.get_stats()}")
    print(f"# Join tuples after shuffle        : {part_partsupp_shufflejoin.get_stats()}")
    print(f"# Join tuples after partsuppagain2 : {partsupp_again.get_stats()}")
    print(f"# min_cost_join                    : {min_cost_join.get_stats()}")
    print(f"# last_join                        : {last_join.get_stats()}")
    if iteration == 0:
        print(f"query,iteration,algo,{reader_wrapper.header_str}")
    print(f"query2,{iteration},{part_partsupp_shufflejoin.name()},{result_str}")
    ######################################################################################
    if os.environ["SHUFFLE_PROBE_TYPE"] != "FULL":
        return

    # Print Aggregation Results:
    groups, aggs = query2_aggregation.get_results()
    torch.set_printoptions(linewidth=200,precision=4,sci_mode=False)
    print(f"# [Local] Result groups: {groups.shape=}")
    print("# p_partkey (first 5)         : ",groups[:5,0])
    print("# min(ps_supplycost) (first 5): ",aggs[:5])

    # Print final sorting results:
    sorted_result = sort_op.get_results()
    print(f"# Sorted results:")
    print("# s_acctbal          : ",sorted_result[0])
    print("# s_suppkey          : ",sorted_result[1])
    print("# ps_partkey         : ",sorted_result[2])

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

