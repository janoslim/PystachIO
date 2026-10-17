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
from modules.morsel_table_module import MorselTableModule
from modules.single_thread_naive import ShuffleWithJoinNaive
from modules.single_thread_shuffle_join_multistream import ShuffleWithJoin_MultiStream
from modules.spilling_aggregation import SpillingAggregation

from modules.query_wrapper import ReaderWrapper
from modules.feed import Feed
from modules.join_wrapper_module import join_wrapper
from modules.aggregation_module import Aggregation,AggregationPostCombine
from modules.pipeline_breaker import PipelineBreaker
from modules.single_node_join import SingleNodeJoin
from .tpch import get_orders,get_customer,get_lineitem
from .dict import *

QUERY18_SQL = """
select
        c_custkey,
        o_orderkey,
        o_totalprice,
        o_orderdate,
        sum(l_quantity)
from
        customer,
        orders,
        lineitem
where
        o_orderkey in (
                select
                        l_orderkey
                from
                        lineitem
                group by
                        l_orderkey
                having
                        sum(l_quantity) > 300
        )
        and c_custkey = o_custkey
        and o_orderkey = l_orderkey
group by
        c_custkey,
        o_orderkey,
        o_totalprice,
        o_orderdate
order by
        c_custkey,
        o_orderkey,
        o_totalprice,
        o_orderdate
"""

def query18_agg(sorted_data,groups_unique,group_idx):
    sum_quantity = sorted_data[0]
    results = torch.zeros(groups_unique.shape[0],dtype=sum_quantity.dtype,device=sorted_data[0].get_device()).scatter_reduce(0, group_idx, sum_quantity, reduce="sum")
    return results

def query18_agg_combine(sorted_data,groups_unique,group_idx):
    results = torch.zeros(groups_unique.shape[0],dtype=sorted_data[1].dtype,device=sorted_data[0].get_device()).scatter_reduce(0, group_idx, sorted_data[1], reduce="sum")
    return results

def query18_having(aggs):
    # having sum(l_quantity) > 300; quantities are DECIMAL(15,2) integers (x100)
    return aggs.gt(30000)


def run_query18_on_parquet(iteration, store, target_device, scale_factor):
    WORLD_SIZE = int(os.environ['WORLD_SIZE'])
    if os.environ["SHUFFLE_TYPE"] == "BLOCKING":
        SHUFFLE_MODULE_TYPE = ShuffleWithJoinNaive
    elif "OVERLAPPED" in os.environ["SHUFFLE_TYPE"] or os.environ["SHUFFLE_TYPE"] == "GRACE":
        # GRACE (SSD spill mode, patch 0003): the join's build side is only the HAVING survivors, so it stays the
        # shuffle join; the pre-aggregation below spills instead
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
                                                    ["l_orderkey", "l_quantity"],
                                                    None,
                                                ),

                                                ("customer", f"{base_folder}/customer.parquet", storage_threads,
                                                    ["c_custkey"],
                                                    None,
                                                ),

                                                ("orders", f"{base_folder}/orders.parquet", storage_threads,
                                                    ["o_orderkey", "o_custkey", "o_totalprice", "o_orderdate"],
                                                    None,
                                                ),
                                        ],
                                read_full_infos = ["customer"]
                                )

    lineitem_reader, customer_reader, orders_reader = reader_wrapper.readers

    if os.environ["HASH_SIZE_NAIVE"] == "True":
        cust_order_hash_init = 160000*scale_factor
        order_lineitem_hash_init = (scale_factor * 1500) // WORLD_SIZE
    else:
        cust_order_hash_init = util.next_prime(160000*scale_factor)
        order_lineitem_hash_init = util.next_prime((scale_factor * 1500) // WORLD_SIZE)

    torch.cuda.empty_cache()
    import gc
    gc.collect()

    customer_orders_join = SingleNodeJoin(target_device, cust_order_hash_init, left_child=customer_reader.get_fifo(),
                                            right_child=orders_reader.get_fifo(), right_col_idx=1)
    shuffle_right_child = customer_orders_join
    # dummy left shuffle child
    shuffle_left_child = MorselTableModule([torch.empty(0,dtype=torch.int64),torch.empty(0,dtype=torch.int64)], -1)
    blocking_aggregation = Feed(lineitem_reader.get_fifo(), target_device)

    # if (tb_size := os.environ.get("SHUFFLE_BUFFER_LEFT")) is not None:
    #     print("# Adding tensor buffer for left join side")
        # co_lineitem_left_child = TensorBuffer(TODO, int(tb_size))
    if (tb_size := os.environ.get("SHUFFLE_BUFFER_RIGHT")) is not None:
        print("# Adding tensor buffer for right join side")
        shuffle_right_child = TensorBuffer(customer_orders_join, int(tb_size))

    # ["c_custkey", "o_orderkey", "o_custkey", "o_totalprice", "o_orderdate"]
    co_lineitem = SHUFFLE_MODULE_TYPE(target_device, order_lineitem_hash_init, shuffle_left_child, 0,
                                      shuffle_right_child, 1, store, store_prefix="")


    if os.environ["SHUFFLE_TYPE"] == "GRACE":
        # SSD spill mode (patch 0003): the partial groups spill by l_orderkey to the data SSD without lineitem and
        # finalize() applies the HAVING per partition, so get_results() holds only the survivors
        query18_preagg_pipeline = SpillingAggregation(query18_agg, query18_agg_combine, [0], projection=[1],
                                                      spill_dirs=os.environ["SPILL_DIR_LINEITEM"],
                                                      post_filter=query18_having,
                                                      # partial groups (l_orderkey, sum(l_quantity)) keep the dtypes
                                                      # of the lineitem columns (l_orderkey, l_quantity)
                                                      partial_dtypes=lineitem_reader.get_dtypes())
    else:
        query18_preagg_pipeline = AggregationPostCombine(query18_agg, query18_agg_combine, [0], projection=[1])
    # ["l_orderkey", "sum_quantity", "c_custkey", "o_orderkey", "o_custkey", "o_totalprice", "o_orderdate"]
    query18_probe_pipeline = AggregationPostCombine(query18_agg, query18_agg_combine, [2,3,5,6], projection=[1])

    ######################################################################################
    def tmp_function():
        agg_finalize_start = time.perf_counter_ns()
        result = blocking_aggregation(probe_pipeline=query18_preagg_pipeline)
        query18_preagg_pipeline.finalize(drop_partial=True)
        # result["time_finalize_agg_ns"] = time.perf_counter_ns() - agg_finalize_start

        groups, aggs = query18_preagg_pipeline.get_results()
        print(f"# preagg groups: {groups.shape}")

        mask = query18_having(aggs)
        groups = torch.masked_select(groups[:,0],mask)
        aggs = torch.masked_select(aggs,mask)

        actual_left_side = MorselTableModule([groups,aggs], 1<<19)
        customer_orders_join.build(customer_reader.get_fifo(), 0)
        customer_orders_join.finish()

        torch.cuda.synchronize(device=target_device)
        print(f"# blocking aggregation took {(time.perf_counter_ns()-agg_finalize_start)/1e6}ms")
        query18_preagg_pipeline.reset()

        # set correct left child:
        co_lineitem.left_child = actual_left_side

        result = co_lineitem(probe_pipeline=query18_probe_pipeline)
        agg_finalize_start = time.perf_counter_ns()
        query18_probe_pipeline.finalize()
        torch.cuda.synchronize(device=target_device)

        result["time_finalize_agg_ns"] = time.perf_counter_ns() - agg_finalize_start
        print(f"# final aggregation took {(result["time_finalize_agg_ns"])/1e6}ms")
        return result

    result_str = reader_wrapper.run(tmp_function)
    print(f"# Tuples after C-O: {customer_orders_join.num_join_tuples}")
    if iteration == 0:
        print(f"query,iteration,algo,{reader_wrapper.header_str}")
    print(f"query18,{iteration},{co_lineitem.name()},{result_str}")
    ######################################################################################
    if os.environ["SHUFFLE_PROBE_TYPE"] != "FULL":
        return
    ######################################################################################
    groups, aggs = query18_probe_pipeline.get_results()
    dump_result({"groups": groups, "aggs": aggs}) # FlowCSD observation, after the timed region
    torch.set_printoptions(linewidth=200,precision=4,sci_mode=False)
    print(f"# [Local] Result groups: {groups.shape=}")

    print("# l_orderkey (first 5)    : ",groups[:5])
    print("# sum_quantity (first 5)  : ",aggs[:5])

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
                final_aggs = torch.zeros(unique_groups.shape[0], dtype=combined_aggs.dtype, device=combined_aggs.device)
                final_aggs.scatter_reduce_(0, group_indices, combined_aggs, reduce="sum")
                print(f"# [Global] Result groups: {unique_groups.shape=}")
                print("# l_orderkey (first 5)    : ",unique_groups[:5])
                print("# sum_quantity (first 5)       : ",final_aggs[:5])

    torch.set_printoptions()



