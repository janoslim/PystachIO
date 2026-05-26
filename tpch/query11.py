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
from modules.pipeline_breaker import PipelineBreaker
from modules.morsel_table_module import MorselTableModule
from modules.tensor_buffer import TensorBuffer
from modules.single_node_join import SingleNodeJoin
from .tpch import get_orders,get_customer,get_lineitem
from .dict import *

# query 11
QUERY11_SQL = """

select
        ps_partkey,
        sum(ps_supplycost * ps_availqty) as value
from
        partsupp,
        supplier,
        nation
where
        ps_suppkey = s_suppkey
        and s_nationkey = n_nationkey
        and n_name = 'GERMANY'
group by
        ps_partkey having
                sum(ps_supplycost * ps_availqty) > (
                        select
                                sum(ps_supplycost * ps_availqty) * 0.0001
                        from
                                partsupp,
                                supplier,
                                nation
                        where
                                ps_suppkey = s_suppkey
                                and s_nationkey = n_nationkey
                                and n_name = 'GERMANY'
                )

"""

def query11_agg(sorted_data,groups_unique,group_idx):
    prod = sorted_data[0] * sorted_data[1]
    results = torch.zeros(groups_unique.shape[0],dtype=sorted_data[0].dtype,device=sorted_data[0].get_device()).scatter_reduce(0, group_idx, prod, reduce="sum", include_self=False)
    return results

def query11_agg_combine(sorted_data,groups_unique,group_idx):
    results = torch.zeros(groups_unique.shape[0],dtype=sorted_data[1].dtype,device=sorted_data[0].get_device()).scatter_reduce(0, group_idx, sorted_data[1], reduce="sum", include_self=False)
    return results

def query11_probe_pipeline(aggregation, joined_tuples):
    # joined_tuples
    # [ 0,          1,            2,           3,            4,            ]
    # ["s_suppkey","s_nationkey","ps_suppkey","ps_availqty","ps_supplycost"]

    aggregation[0] += (joined_tuples[3] * joined_tuples[4]).sum()


def run_query11_on_parquet(iteration, store, target_device, scale_factor):
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

                                                ("supplier", f"{base_folder}/supplier.parquet", storage_threads,
                                                    ["s_suppkey","s_nationkey"],
                                                    TableFilter({
                                                        1: [FilterCondition(FilterOp.EQ, 7)]
                                                    })
                                                ),

                                                ("partsupp", f"{base_folder}/partsupp.parquet", storage_threads,
                                                    ["ps_suppkey","ps_availqty","ps_supplycost"],
                                                    None
                                                ),

                                                ("supplier2", f"{base_folder}/supplier.parquet", storage_threads,
                                                    ["s_suppkey","s_nationkey"],
                                                    TableFilter({
                                                        1: [FilterCondition(FilterOp.EQ, 7)]
                                                    })
                                                ),

                                                ("partsupp2", f"{base_folder}/partsupp.parquet", storage_threads,
                                                    ["ps_partkey","ps_suppkey","ps_availqty","ps_supplycost"],
                                                    None
                                                ),

                                        ],
                                read_full_infos = []
                                )

    supplier_reader, partsupp_reader, supplier_reader2, partsupp_reader2 = reader_wrapper.readers

    if os.environ["HASH_SIZE_NAIVE"] == "True":
        supplier_hash_init = 5000*scale_factor
    else:
        supplier_hash_init = util.next_prime(5000*scale_factor)

    torch.cuda.empty_cache()
    import gc
    gc.collect()

    shuffle_left_child = supplier_reader.get_fifo()
    shuffle2_left_child = supplier_reader2.get_fifo()
    shuffle_right_child = partsupp_reader.get_fifo()
    shuffle2_right_child = partsupp_reader2.get_fifo()

    if (tb_size := os.environ.get("SHUFFLE_BUFFER_LEFT")) is not None:
        print("# Adding tensor buffer for left join side")
        shuffle_left_child = TensorBuffer(supplier_reader.get_fifo(), int(tb_size))
        shuffle2_left_child = TensorBuffer(supplier_reader2.get_fifo(), int(tb_size))
    if (tb_size := os.environ.get("SHUFFLE_BUFFER_RIGHT")) is not None:
        print("# Adding tensor buffer for right join side")
        shuffle_right_child = TensorBuffer(partsupp_reader.get_fifo(), int(tb_size))
        shuffle2_right_child = TensorBuffer(partsupp_reader2.get_fifo(), int(tb_size))

    first_shufflejoin = SHUFFLE_MODULE_TYPE(target_device, supplier_hash_init, shuffle_left_child, 0,
                                                    shuffle_right_child, 0, store, store_prefix="")
    second_shufflejoin = SHUFFLE_MODULE_TYPE(target_device, supplier_hash_init, shuffle2_left_child, 0,
                                                    shuffle2_right_child, 1, store, store_prefix="")
    # joined_tuples
    # [ 0,          1,            2,           3,            4,            ]
    # ["s_suppkey","s_nationkey","ps_suppkey","ps_availqty","ps_supplycost"]
    query11_aggregation = torch.zeros(1,dtype=torch.int64,device="cuda:0")
    probe_pipeline = partial(query11_probe_pipeline, query11_aggregation)

    # [ 0,          1,            2,           3,           4,            5             ]
    # ["s_suppkey","s_nationkey","ps_partkey","ps_suppkey","ps_availqty","ps_supplycost"]
    query11_second_aggregation = AggregationPostCombine(query11_agg, query11_agg_combine, [2], projection=[4,5])

    ######################################################################################
    def tmp_function():

        result = first_shufflejoin(probe_pipeline=probe_pipeline)
        agg_finalize_start = time.perf_counter_ns()
        dist.all_reduce(query11_aggregation, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize(device=target_device)

        result2 = second_shufflejoin(probe_pipeline=query11_second_aggregation)
        query11_second_aggregation.finalize()
        partkey, sum_pscost = query11_second_aggregation.get_results()
        mask = torch.gt(sum_pscost, torch.div(query11_aggregation,10000,rounding_mode='floor').item())
        as_indices = mask.nonzero().squeeze()

        after_having = torch.index_select(partkey,0,as_indices)
        # query 11 is empty at e.g. sf 100
        torch.cuda.synchronize(device=target_device)
        print(f"# After having shape: {after_having.shape}, first five: {after_having[:5]}")

        combined_result = {k: result.get(k, 0) + result2.get(k, 0) for k in set(result) | set(result2)}
        return combined_result

    result_str = reader_wrapper.run(tmp_function)
    # print(f"# nation_supplier_join tuples      : {nation_supplier_join.num_join_tuples}")
    # print(f"# last_join                        : {last_join.num_join_tuples}")
    # print(f"# min_cost_join                    : {min_cost_join.num_join_tuples}")
    print(f"# Join tuples after shuffle         : {first_shufflejoin._matched_pairs.sum().cpu().item()}")
    print(f"# Join tuples after 2ndshuffle      : {second_shufflejoin._matched_pairs.sum().cpu().item()}")
    # print(f"# Join tuples after shuffle2       : {part_partsupp_shufflejoin2._matched_pairs.sum().cpu().item()}")
    if iteration == 0:
        print(f"query,iteration,algo,{reader_wrapper.header_str}")
    print(f"query11,{iteration},{first_shufflejoin.name()},{result_str}")
    ######################################################################################
    if os.environ["SHUFFLE_PROBE_TYPE"] != "FULL":
        return

    # Print Aggregation Results:
    print(f"# Sum: {query11_aggregation}")
    print(f"# Sum*0.0001: {torch.div(query11_aggregation,10000,rounding_mode='floor')}")
    groups, aggs = query11_second_aggregation.get_results()
    torch.set_printoptions(linewidth=200,precision=4,sci_mode=False)
    print(f"# [Local] Result groups: {groups.shape=}")
    print("# p_partkey (first 5)                        : ",groups[:5,0])
    print("# sum(ps_supplycost*ps_availqty) (first 5)   : ",aggs[:5])
    

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

