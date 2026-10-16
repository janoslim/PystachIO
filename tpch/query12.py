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
from modules.filter_module import FilterCondition, FilterOp, FilterOpCol, TableFilter
from modules.single_thread_naive import ShuffleWithJoinNaive
from modules.single_thread_shuffle_join_multistream import ShuffleWithJoin_MultiStream

from modules.query_wrapper import ReaderWrapper
from modules.join_wrapper_module import join_wrapper
from modules.aggregation_module import Aggregation,AggregationPostCombine
from modules.tensor_buffer import TensorBuffer
from modules.pipeline_breaker import PipelineBreaker
from modules.single_node_join import SingleNodeJoin
from .tpch import get_orders,get_customer,get_lineitem
from .dict import *

# sf400:
# 12462614 of 2400012063 lineitems qualify (0.0052)
# vs 600000000 orders
QUERY12_SQL = """
select
        l_shipmode,
        sum(case
                when o_orderpriority = '1-URGENT'
                        or o_orderpriority = '2-HIGH'
                        then 1
                else 0
        end) as high_line_count,
        sum(case
                when o_orderpriority <> '1-URGENT'
                        and o_orderpriority <> '2-HIGH'
                        then 1
                else 0
        end) as low_line_count
from
        orders,
        lineitem
where
        o_orderkey = l_orderkey
        and l_shipmode in ('MAIL', 'SHIP')
        and l_commitdate < l_receiptdate
        and l_shipdate < l_commitdate
        and l_receiptdate >= date '1994-01-01'
        and l_receiptdate < date '1994-01-01' + interval '1' year
group by
        l_shipmode
order by
        l_shipmode;
"""

def query12_agg(sorted_data,groups_unique,group_idx):
    highline = (sorted_data[0] < 2).to(torch.int64) # hardcoded "3-MEDIUM"
    lowline = (sorted_data[0] >= 2).to(torch.int64)

    results = torch.zeros((groups_unique.shape[0], 2),dtype=torch.int64,device=sorted_data[0].get_device())
    results[:,0].scatter_reduce_(0, group_idx, highline, reduce="sum")
    results[:,1].scatter_reduce_(0, group_idx, lowline, reduce="sum")
    return results

def query12_agg_combine(sorted_data,groups_unique,group_idx):
    results = torch.zeros((groups_unique.shape[0],2),dtype=sorted_data[1].dtype,device=sorted_data[0].get_device())

    results[:,0].scatter_reduce_(0, group_idx, sorted_data[1][:,0], reduce="sum")
    results[:,1].scatter_reduce_(0, group_idx, sorted_data[1][:,1], reduce="sum")
    return results


def run_query12_on_parquet(iteration, store, target_device, scale_factor):
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

                                                ("orders", f"{base_folder}/orders.parquet", storage_threads,
                                                    ["o_orderkey", "o_orderpriority"],
                                                    None
                                                ),

                                                # ("lineitem", f"{page_base_folder}/lineitem_10000000_pc{PAGE}.parquet", storage_threads,
                                                ("lineitem", f"{base_folder}/lineitem.parquet", storage_threads,
                                                    ["l_orderkey", "l_shipdate", "l_commitdate", "l_receiptdate","l_shipmode"],
                                                    TableFilter({
                                                        1: [FilterCondition(FilterOpCol.LTC, 2)],  # l_shipdate < l_commitdate
                                                        2: [FilterCondition(FilterOpCol.LTC, 3)],  # l_commitdate < l_receiptdate
                                                        3: [FilterCondition(FilterOp.GE, datetime.date(1994, 1, 1)),    # l_receiptdate >= date '1994-01-01'
                                                            FilterCondition(FilterOp.LT, datetime.date(1995, 1, 1))],   # l_receiptdate < date '1995-01-01'
                                                        4: [FilterCondition(FilterOp.II, torch.tensor([
                                                                                                      string_to_categoryidx("l_shipmode","MAIL"),
                                                                                                      string_to_categoryidx("l_shipmode","SHIP")
                                                                                                      ],device=target_device))]
                                                    })
                                                )
                                        ]
                                )

    orders_reader, lineitem_reader = reader_wrapper.readers

    if os.environ["HASH_SIZE_NAIVE"] == "True":
        order_lineitem_hash_init = (scale_factor * 100000) // WORLD_SIZE
    else:
        order_lineitem_hash_init = util.next_prime((scale_factor * 100000) // WORLD_SIZE)

    torch.cuda.empty_cache()
    import gc
    gc.collect()

    shuffle_left_child = lineitem_reader.get_fifo()
    shuffle_right_child = orders_reader.get_fifo()

    if (tb_size := os.environ.get("SHUFFLE_BUFFER_LEFT")) is not None:
        print("# Adding tensor buffer for left join side")
        shuffle_left_child = TensorBuffer(lineitem_reader.get_fifo(), int(tb_size))
    if (tb_size := os.environ.get("SHUFFLE_BUFFER_RIGHT")) is not None:
        print("# Adding tensor buffer for right join side")
        shuffle_right_child = TensorBuffer(orders_reader.get_fifo(), int(tb_size))

    co_lineitem = SHUFFLE_MODULE_TYPE(target_device, order_lineitem_hash_init, shuffle_left_child, 0,
                                      shuffle_right_child, 0, store, store_prefix="")
    # joined_tuples
    # [ 0,            1,            2,              3,              4,           5,            6]
    # ["l_orderkey", "l_shipdate", "l_commitdate", "l_receiptdate","l_shipmode","o_orderkey", "o_orderpriority"]
    query12_probe_pipeline = AggregationPostCombine(query12_agg, query12_agg_combine, [4], projection=[6])

    ######################################################################################
    def tmp_function():
        result = co_lineitem(probe_pipeline=query12_probe_pipeline)
        agg_finalize_start = time.perf_counter_ns()
        query12_probe_pipeline.finalize()
        torch.cuda.synchronize(device=target_device)

        result["time_finalize_agg_ns"] = time.perf_counter_ns() - agg_finalize_start

        return result

    result_str = reader_wrapper.run(tmp_function)
    if iteration == 0:
        print(f"query,iteration,algo,{reader_wrapper.header_str}")
    print(f"query12,{iteration},{co_lineitem.name()},{result_str}")
    ######################################################################################

    if os.environ["SHUFFLE_PROBE_TYPE"] != "FULL":
        return

    ######################################################################################

    groups, aggs = query12_probe_pipeline.get_results()
    dump_result({"groups": groups, "aggs": aggs}) # FlowCSD observation, after the timed region

    torch.set_printoptions(linewidth=200,precision=4,sci_mode=False)
    print(f"# [Local] Result groups: {groups.shape=}")
    print("# l_shipmode     : ",[categoryidx_to_string("l_shipmode",x) for x in groups[:,0].tolist()])
    print("# Highline Count : ",aggs[:,0])
    print("# Lowline  Count : ",aggs[:,1])

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
                final_aggs = torch.zeros((unique_groups.shape[0], 2), dtype=combined_aggs.dtype, device=combined_aggs.device)
                final_aggs[:,0].scatter_reduce_(0, group_indices, combined_aggs[:,0], reduce="sum")
                final_aggs[:,1].scatter_reduce_(0, group_indices, combined_aggs[:,1], reduce="sum")
                print(f"# [Global] Result groups: {unique_groups.shape=}")
                print("# l_shipmode     : ",[categoryidx_to_string("l_shipmode",x) for x in unique_groups[:,0].tolist()])
                print("# Highline Count : ",final_aggs[:,0])
                print("# Lowline  Count : ",final_aggs[:,1])

    torch.set_printoptions()
