import sys
import os
import time
import datetime
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
import pandas as pd
import torch
import torch.distributed as dist

from helpers import util
from modules.filter_module import FilterCondition, FilterOp, FilterOpCol, TableFilter
from modules.single_thread_naive import ShuffleWithJoinNaive
from modules.single_thread_shuffle_join_multistream import ShuffleWithJoin_MultiStream

from modules.query_wrapper import ReaderWrapper
from modules.join_wrapper_module import join_wrapper
from modules.aggregation_module import Aggregation,AggregationPostCombine,group_by
from modules.tensor_buffer import TensorBuffer
from modules.pipeline_breaker import PipelineBreaker
from modules.single_node_join import SingleNodeJoin
from .tpch import get_orders,get_customer,get_lineitem
from .dict import *

# sf400:
QUERY13_SQL = """
select
        c_count,
        count(*) as custdist
from
        (
                select
                        c_custkey,
                        count(o_orderkey)
                from
                        customer left outer join orders on
                                c_custkey = o_custkey
                group by
                        c_custkey
        ) as c_orders (c_custkey, c_count)
group by
        c_count
order by
        c_count

"""

def query13_agg(sorted_data,groups_unique,group_idx):
    ones = torch.ones(sorted_data[0].shape[0],dtype=torch.int64,device=sorted_data[0].get_device())
    results = torch.zeros(groups_unique.shape[0],dtype=torch.int64,device=sorted_data[0].get_device()).scatter_reduce(0, group_idx, ones, reduce="sum")
    return results

def query13_agg_combine(sorted_data,groups_unique,group_idx):
    results = torch.zeros(groups_unique.shape[0],dtype=sorted_data[1].dtype,device=sorted_data[0].get_device()).scatter_reduce(0, group_idx, sorted_data[1], reduce="sum")
    return results


def run_query13_on_parquet(iteration, store, target_device, scale_factor):
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
                                                ("customer", f"{base_folder}/customer.parquet", storage_threads,
                                                    ["c_custkey"],
                                                    None
                                                ),

                                                ("orders", f"{base_folder}/orders.parquet", storage_threads,
                                                    ["o_orderkey","o_custkey"],
                                                    None
                                                ),

                                        ]
                                )

    customer_reader,orders_reader = reader_wrapper.readers

    if os.environ["HASH_SIZE_NAIVE"] == "True":
        customer_order_hash_init = (scale_factor * 160000) // WORLD_SIZE
    else:
        customer_order_hash_init = util.next_prime((scale_factor * 160000) // WORLD_SIZE)

    torch.cuda.empty_cache()
    import gc
    gc.collect()

    shuffle_left_child = customer_reader.get_fifo()
    shuffle_right_child = orders_reader.get_fifo()

    if (tb_size := os.environ.get("SHUFFLE_BUFFER_LEFT")) is not None:
        print("# Adding tensor buffer for left join side")
        shuffle_left_child = TensorBuffer(customer_reader.get_fifo(), int(tb_size))
    if (tb_size := os.environ.get("SHUFFLE_BUFFER_RIGHT")) is not None:
        print("# Adding tensor buffer for right join side")
        shuffle_right_child = TensorBuffer(orders_reader.get_fifo(), int(tb_size))

    cust_order_shuffle = SHUFFLE_MODULE_TYPE(target_device, customer_order_hash_init, shuffle_left_child, 0,
                                      shuffle_right_child, 1, store, store_prefix="", keep_build_side=True)
    # joined_tuples
    # [ 0,           1,           2]
    # ["c_custkey", "o_orderkey","o_custkey"]
    query13_probe_pipeline = AggregationPostCombine(query13_agg, query13_agg_combine, [0], projection=[0])

    ######################################################################################
    def tmp_function():
        result = cust_order_shuffle(probe_pipeline=query13_probe_pipeline)
        agg_finalize_start = time.perf_counter_ns()
        query13_probe_pipeline.finalize()
        torch.cuda.synchronize(device=target_device)
        groups, aggs = query13_probe_pipeline.get_results()
        # 1) left outer join
        full_groups = torch.concat((groups[:,0],cust_order_shuffle.build_side_cols[0]))
        full_aggs = torch.concat((aggs,torch.zeros_like(cust_order_shuffle.build_side_cols[0])))
        sorted_data, groups_unique, group_idx = group_by([full_groups,full_aggs],[0],projection=[1])
        # print(f"{sorted_data[0].shape=},{groups_unique.shape=},{group_idx.shape=}")
        # simple sum:
        results = torch.zeros(groups_unique.shape[0],dtype=sorted_data[0].dtype,device=sorted_data[0].get_device()).scatter_reduce(0, group_idx, sorted_data[0], reduce="sum")
        print("# c_custkey     : ",groups_unique[:8,0])
        print("# order_count   : ",results[:8])

        # 2) group by order count
        sorted_data, groups_unique, group_idx = group_by([results],[0],projection=[0])
        ones = torch.ones(sorted_data[0].shape[0],dtype=torch.int64,device=sorted_data[0].get_device())
        final_results = torch.zeros(groups_unique.shape[0],dtype=sorted_data[0].dtype,device=sorted_data[0].get_device()).scatter_reduce(0, group_idx, ones, reduce="sum")
        print("# order_count     : ",groups_unique[:8,0])
        print("# customers       : ",final_results[:8])

        result["time_finalize_agg_ns"] = time.perf_counter_ns() - agg_finalize_start

        return result

    result_str = reader_wrapper.run(tmp_function)
    if iteration == 0:
        print(f"query,iteration,algo,{reader_wrapper.header_str}")
    print(f"query13,{iteration},{cust_order_shuffle.name()},{result_str}")
    ######################################################################################

    if os.environ["SHUFFLE_PROBE_TYPE"] != "FULL":
        return

    ######################################################################################

    # groups, aggs = query13_probe_pipeline.get_results()

    # torch.set_printoptions(linewidth=200,precision=4,sci_mode=False)
    # print(f"# [Local] Result groups: {groups.shape=}")
    # print("# c_custkey     : ",groups[:5,0])
    # print("# Order count   : ",aggs[:5])

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
    #             final_aggs = torch.zeros((unique_groups.shape[0], 2), dtype=combined_aggs.dtype, device=combined_aggs.device)
    #             final_aggs[:,0].scatter_reduce_(0, group_indices, combined_aggs[:,0], reduce="sum")
    #             final_aggs[:,1].scatter_reduce_(0, group_indices, combined_aggs[:,1], reduce="sum")
    #             print(f"# [Global] Result groups: {unique_groups.shape=}")
    #             print("# c_custkey     : ",unique_groups[:5,0])
    #             print("# Order count   : ",final_aggs[:5])

    torch.set_printoptions()
