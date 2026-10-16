import sys
import os
import time
import datetime
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
import pandas as pd
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
QUERY1_SQL = """
select
        l_returnflag,
        l_linestatus,
        sum(l_quantity) as sum_qty,
        sum(l_extendedprice) as sum_base_price,
        sum(l_extendedprice * (1 - l_discount)) as sum_disc_price,
        sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) as sum_charge,
        count(*) as count_order
from
        lineitem
where
        l_shipdate <= date '1998-12-01' - interval '90' day
group by
        l_returnflag,
        l_linestatus
order by
        l_returnflag,
        l_linestatus;
"""

def query1_agg(sorted_data,groups_unique,group_idx):
    # sum(l_extendedprice * (1 - l_discount)) as revenue
    # since the scale is 2, 100 equals a value of 1:
    sum_disc_price = sorted_data[1] * (100 - sorted_data[2])

    results = torch.zeros((groups_unique.shape[0], 5),dtype=torch.int64,device=sorted_data[0].get_device())
    results[:,0].scatter_reduce_(0, group_idx, sorted_data[0], reduce="sum") # sum_quantity
    results[:,1].scatter_reduce_(0, group_idx, sorted_data[1], reduce="sum") # sum_base_price
    results[:,2].scatter_reduce_(0, group_idx, sum_disc_price, reduce="sum") # sum_disc_price
    results[:,3].scatter_reduce_(0, group_idx, sum_disc_price*(100 + sorted_data[3]), reduce="sum") # sum_charge
    results[:,4].scatter_reduce_(0, group_idx, torch.ones(group_idx.shape[0],dtype=torch.int64,device=sorted_data[0].get_device()), reduce="sum") # count_order

    return results

def query1_agg_combine(sorted_data,groups_unique,group_idx):
    results = torch.zeros((groups_unique.shape[0],5),dtype=sorted_data[1].dtype,device=sorted_data[0].get_device())

    results[:,0].scatter_reduce_(0, group_idx, sorted_data[1][:,0], reduce="sum")
    results[:,1].scatter_reduce_(0, group_idx, sorted_data[1][:,1], reduce="sum")
    results[:,2].scatter_reduce_(0, group_idx, sorted_data[1][:,2], reduce="sum")
    results[:,3].scatter_reduce_(0, group_idx, sorted_data[1][:,3], reduce="sum")
    results[:,4].scatter_reduce_(0, group_idx, sorted_data[1][:,4], reduce="sum")
    return results


def run_query1_on_parquet(iteration, store, target_device, scale_factor):
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
                                                    ["l_quantity","l_extendedprice","l_discount","l_tax",
                                                    "l_returnflag","l_linestatus", "l_shipdate"],
                                                    TableFilter({
                                                        6: [FilterCondition(FilterOp.LE, datetime.date(1998, 9, 2))]
                                                        # 6: [FilterCondition(FilterOp.GE, datetime.date(1998, 9, 2))]
                                                    })
                                                    # None
                                                )
                                        ]
                                )

    lineitem_reader = reader_wrapper.readers[0]

    torch.cuda.empty_cache()
    import gc
    gc.collect()

    root_operator = Feed(lineitem_reader.get_fifo(), target_device)
    # joined_tuples
    # [ 0,           1,                2,           3,      4,             5,             6]
    # ["l_quantity","l_extendedprice","l_discount","l_tax","l_returnflag","l_linestatus","l_shipdate"]
    query1_probe_pipeline = AggregationPostCombine(query1_agg, query1_agg_combine, [4,5], projection=[0,1,2,3])

    ######################################################################################
    def tmp_function():
        result = root_operator(probe_pipeline=query1_probe_pipeline)
        agg_finalize_start = time.perf_counter_ns()
        query1_probe_pipeline.finalize()
        torch.cuda.synchronize(device=target_device)

        result["time_finalize_agg_ns"] = time.perf_counter_ns() - agg_finalize_start
        return result

    result_str = reader_wrapper.run(tmp_function)
    print(f"# Tuples local: {root_operator.seen_tuples}")
    if iteration == 0:
        print(f"query,iteration,algo,{reader_wrapper.header_str}")
    print(f"query1,{iteration},{root_operator.name()},{result_str}")
    ######################################################################################

    if os.environ["SHUFFLE_PROBE_TYPE"] != "FULL":
        return

    ######################################################################################
    groups, aggs = query1_probe_pipeline.get_results()
    dump_result({"groups": groups, "aggs": aggs}) # FlowCSD observation, after the timed region

    torch.set_printoptions(linewidth=200,precision=4,sci_mode=False)
    print(f"# [Local] Result groups: {groups.shape=}")
    print("# l_returnflag   : ", [categoryidx_to_string("l_returnflag",x) for x in groups[:,0].tolist()])
    print("# l_linestatus   : ", [categoryidx_to_string("l_linestatus",x) for x in groups[:,1].tolist()])
    print("# sum_qty        : ",aggs[:,0])
    print("# sum_base_price : ",aggs[:,1])
    print("# sum_disc_price : ",aggs[:,2])
    print("# sum_charge     : ",aggs[:,3])
    print("# count_order    : ",aggs[:,4])
    
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
                final_aggs = torch.zeros((unique_groups.shape[0], 5), dtype=combined_aggs.dtype, device=combined_aggs.device)
                # final_aggs = torch.zeros(unique_groups.shape[0], dtype=combined_aggs.dtype, device=combined_aggs.device)
                # final_aggs.scatter_reduce_(0, group_indices, combined_aggs, reduce="sum")
                final_aggs[:,0].scatter_reduce_(0, group_indices, combined_aggs[:,0], reduce="sum")
                final_aggs[:,1].scatter_reduce_(0, group_indices, combined_aggs[:,1], reduce="sum")
                final_aggs[:,2].scatter_reduce_(0, group_indices, combined_aggs[:,2], reduce="sum")
                final_aggs[:,3].scatter_reduce_(0, group_indices, combined_aggs[:,3], reduce="sum")
                final_aggs[:,4].scatter_reduce_(0, group_indices, combined_aggs[:,4], reduce="sum")
                print(f"# [Global] Result groups: {unique_groups.shape=}")
                print("# l_returnflag   : ", [categoryidx_to_string("l_returnflag",x) for x in unique_groups[:,0].tolist()])
                print("# l_linestatus   : ", [categoryidx_to_string("l_linestatus",x) for x in unique_groups[:,1].tolist()])
                print("# sum_qty        : ",final_aggs[:,0])
                print("# sum_base_price : ",final_aggs[:,1])
                print("# sum_disc_price : ",final_aggs[:,2])
                print("# sum_charge     : ",final_aggs[:,3])
                print("# count_order    : ",final_aggs[:,4])

    torch.set_printoptions()

