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
from modules.aggregation_module import Aggregation,AggregationPostCombine
from modules.pipeline_breaker import PipelineBreaker
from modules.tensor_buffer import TensorBuffer
from modules.distinct import Distinct
from modules.single_node_join import SingleNodeJoin
from .tpch import get_orders,get_customer,get_lineitem
from .dict import *

# sf400:
# 550188756distinct 1517335695total of 2400012063 lineitems qualify (0.6322200285540814)
# vs 22945702 of 600000000 orders
QUERY4_SQL = """
select
        o_orderpriority,
        count(*) as order_count
from
        orders
where
        o_orderdate >= date '1993-07-01'
        and o_orderdate < date '1993-07-01' + interval '3' month
        and exists (
                select
                        *
                from
                        lineitem
                where
                        l_orderkey = o_orderkey
                        and l_commitdate < l_receiptdate
        )
group by
        o_orderpriority
order by
        o_orderpriority
"""

QUERY4_SQL_UNNESTED = """
with temp as (
  select distinct(l_orderkey) as l_orderkey
  from lineitem
  where l_commitdate < l_receiptdate
)

select
    o.o_orderpriority,
    count(*) as order_count
from
    orders o, temp
where
    o.o_orderdate >= date '1993-07-01'
    and o.o_orderdate < date '1993-07-01' + interval '3' month
    and temp.l_orderkey = o.o_orderkey
group by
    o.o_orderpriority
order by
    o.o_orderpriority;
"""

def query4_agg(sorted_data,groups_unique,group_idx):
    ones = torch.ones(sorted_data[0].shape[0],dtype=torch.int64,device=sorted_data[0].get_device())
    results = torch.zeros(groups_unique.shape[0],dtype=torch.int64,device=sorted_data[0].get_device()).scatter_reduce(0, group_idx, ones, reduce="sum")
    return results

def query4_agg_combine(sorted_data,groups_unique,group_idx):
    results = torch.zeros(groups_unique.shape[0],dtype=sorted_data[1].dtype,device=sorted_data[0].get_device()).scatter_reduce(0, group_idx, sorted_data[1], reduce="sum")
    return results


def run_query4_on_parquet(iteration, store, target_device, scale_factor):
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
                                                    ["o_orderkey", "o_orderdate", "o_orderpriority"],
                                                    TableFilter({
                                                        1: [FilterCondition(FilterOp.GE, datetime.date(1993,7,1)),
                                                            FilterCondition(FilterOp.LT, datetime.date(1993,10,1))] # o_orderdate
                                                    })
                                                ),

                                                ("lineitem", f"{base_folder}/lineitem.parquet", storage_threads,
                                                    ["l_orderkey", "l_commitdate", "l_receiptdate"],
                                                    TableFilter({
                                                        1: [FilterCondition(FilterOpCol.LTC, 2)] # l_commitdate < l_receiptdate
                                                    })
                                                ),
                                        ]
                                )

    orders_reader, lineitem_reader = reader_wrapper.readers

    if os.environ["HASH_SIZE_NAIVE"] == "True":
        order_lineitem_hash_init = (scale_factor * 800000) // WORLD_SIZE
    else:
        order_lineitem_hash_init = util.next_prime((scale_factor * 800000) // WORLD_SIZE)

    torch.cuda.empty_cache()
    import gc
    gc.collect()

    pre_shuffle_distinct = Distinct(lineitem_reader.get_fifo(),0)

    shuffle_left_child = pre_shuffle_distinct
    shuffle_right_child = orders_reader.get_fifo()

    if (tb_size := os.environ.get("SHUFFLE_BUFFER_LEFT")) is not None:
        print("# Adding tensor buffer for left join side")
        shuffle_left_child = TensorBuffer(pre_shuffle_distinct, int(tb_size))
    if (tb_size := os.environ.get("SHUFFLE_BUFFER_RIGHT")) is not None:
        print("# Adding tensor buffer for right join side")
        shuffle_right_child = TensorBuffer(orders_reader.get_fifo(), int(tb_size))

    shuffle = SHUFFLE_MODULE_TYPE(target_device, order_lineitem_hash_init, shuffle_left_child, 0,
                                      shuffle_right_child, 0, store, store_prefix="", distinct_ht=True)
    # joined_tuples
    # [ 0,            1,            2,             3]
    # ["l_orderkey", "o_orderkey", "o_orderdate", "o_orderpriority",]
    query4_probe_pipeline = AggregationPostCombine(query4_agg, query4_agg_combine, [3], projection=[0])

    ######################################################################################
    def tmp_function():
        result = shuffle(probe_pipeline=query4_probe_pipeline)

        agg_finalize_start = time.perf_counter_ns()
        query4_probe_pipeline.finalize()
        torch.cuda.synchronize(device=target_device)

        result["time_finalize_agg_ns"] = time.perf_counter_ns() - agg_finalize_start
        return result

    result_str = reader_wrapper.run(tmp_function)
    if iteration == 0:
        print(f"query,iteration,algo,{reader_wrapper.header_str}")
    print(f"query4,{iteration},{shuffle.name()},{result_str}")
    ######################################################################################

    if os.environ["SHUFFLE_PROBE_TYPE"] != "FULL":
        return

    ######################################################################################
    groups, aggs = query4_probe_pipeline.get_results()
    from helpers.flowcsd_observe import dump_result
    dump_result({"groups": groups, "aggs": aggs})

    torch.set_printoptions(linewidth=200,precision=4,sci_mode=False)
    print(f"# [Local] Result groups: {groups.shape=}")
    print("# o_orderpriority ", [categoryidx_to_string("o_orderpriority",x) for x in groups[:,0].tolist()])
    print("# count          : ",aggs)
    
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
                print("# o_orderpriority ", [categoryidx_to_string("o_orderpriority",x) for x in unique_groups[:,0].tolist()])
                print("# count          : ",final_aggs)

    torch.set_printoptions()

