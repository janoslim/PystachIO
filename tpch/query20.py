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
from modules.tensor_buffer import TensorBuffer
from modules.single_node_join import SingleNodeJoin
from .tpch import get_orders,get_customer,get_lineitem
from .dict import *

QUERY20_SQL = """
select
        s_suppkey,
        s_nationkey
from
        supplier
where
        s_suppkey in (
                select
                        ps_suppkey
                from
                        partsupp
                where
                        ps_partkey in (
                                select
                                        p_partkey
                                from
                                        part
                                where
                                        p_size = 1
                        )
                        and ps_availqty > (
                                select
                                        0.5 * sum(l_quantity)
                                from
                                        lineitem
                                where
                                        l_partkey = ps_partkey
                                        and l_suppkey = ps_suppkey
                                        and l_shipdate >= date '1994-01-01'
                                        and l_shipdate < date '1994-01-01' + interval '1' year
                        )
        )
        and s_nationkey = 3
order by
        s_suppkey
"""

def query20_agg(sorted_data,groups_unique,group_idx):
    # [ 0 ]
    # ["l_quantity"]
    results = torch.zeros(groups_unique.shape[0],dtype=sorted_data[0].dtype,device=sorted_data[0].get_device()).scatter_reduce(0, group_idx, sorted_data[0], reduce="sum", include_self=False)
    return results

def query20_agg_combine(sorted_data,groups_unique,group_idx):
    results = torch.zeros(groups_unique.shape[0],dtype=sorted_data[1].dtype,device=sorted_data[0].get_device()).scatter_reduce(0, group_idx, sorted_data[1], reduce="sum", include_self=False)
    return results


def run_query20_on_parquet(iteration, store, target_device, scale_factor):
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
    # {'ps_availqty', 'l_suppkey', 'n_nationkey', 'ps_partkey', 'l_shipdate', 'n_name', 'ps_suppkey', 'p_name', 'p_partkey', 'l_quantity', 'l_partkey'}
    reader_wrapper = ReaderWrapper(store, target_device,
                                         # name, filepath, threads, cols, filters
                                table_infos= [  
                                                ("supplier", f"{base_folder}/supplier.parquet", storage_threads,
                                                    ["s_suppkey", "s_nationkey"],
                                                    TableFilter({
                                                        1: [FilterCondition(FilterOp.EQ, string_to_categoryidx("n_name","CANADA"))], # nationkey= CANADA
                                                    })
                                                ),

                                                ("part", f"{base_folder}/part.parquet", storage_threads,
                                                    ["p_partkey", "p_size"],
                                                    TableFilter({
                                                        1: [FilterCondition(FilterOp.EQ, 1)], # p_size
                                                    })
                                                ),

                                                ("partsupp", f"{base_folder}/partsupp.parquet", storage_threads,
                                                    ["ps_partkey", "ps_suppkey", "ps_availqty"],
                                                    None
                                                ),

                                                ("lineitem", f"{base_folder}/lineitem.parquet", storage_threads,
                                                    ["l_partkey", "l_suppkey", "l_quantity", "l_shipdate"],
                                                    TableFilter({
                                                        3: [FilterCondition(FilterOp.GE, datetime.date(1994, 1, 1)),
                                                            FilterCondition(FilterOp.LT, datetime.date(1995, 1, 1))]  # l_shipdate
                                                    })
                                                ),

                                        ]
                                )

    supplier_reader, part_reader, partsupp_reader, lineitem_reader = reader_wrapper.readers

    if os.environ["HASH_SIZE_NAIVE"] == "True":
        supplier_hash_init = (scale_factor * 1000) // WORLD_SIZE
        part_hash_init = (scale_factor * 5000) // WORLD_SIZE
        pps_l_hash_init = (scale_factor * 1000) // WORLD_SIZE
    else:
        supplier_hash_init = util.next_prime((scale_factor * 1000) // WORLD_SIZE)
        part_hash_init = util.next_prime((scale_factor * 5000) // WORLD_SIZE)
        pps_l_hash_init = util.next_prime((scale_factor * 1000) // WORLD_SIZE)

    torch.cuda.empty_cache()
    import gc
    gc.collect()

    supplier_join = SingleNodeJoin(target_device, supplier_hash_init, left_child=supplier_reader.get_fifo())

    p_ps_join = SingleNodeJoin(target_device, part_hash_init, left_child=part_reader.get_fifo(),
                                            right_child=partsupp_reader.get_fifo(), right_col_idx=0)

    shuffle_left_child = p_ps_join
    shuffle_right_child = lineitem_reader.get_fifo()

    if (tb_size := os.environ.get("SHUFFLE_BUFFER_LEFT")) is not None:
        print("# Adding tensor buffer for left join side")
        shuffle_left_child = TensorBuffer(p_ps_join, int(tb_size))
    if (tb_size := os.environ.get("SHUFFLE_BUFFER_RIGHT")) is not None:
        print("# Adding tensor buffer for right join side")
        shuffle_right_child = TensorBuffer(lineitem_reader.get_fifo(), int(tb_size))

    ps_l_shuffle = SHUFFLE_MODULE_TYPE(target_device, pps_l_hash_init, shuffle_left_child, 2,
                                      shuffle_right_child, 0, store, store_prefix="", add_join_attrib=[(3,1)])
    # joined_tuples

    query20_probe_pipeline = AggregationPostCombine(query20_agg, query20_agg_combine, [3], projection=[7])
    # query20_aggregation = torch.zeros(1,dtype=torch.int64,device="cuda:0")
    # probe_pipeline = partial(query20_probe_pipeline, query20_aggregation)

    ######################################################################################
    def tmp_function():
        supplier_join.build(supplier_reader.get_fifo(), 0)
        p_ps_join.build(part_reader.get_fifo(), 0)
        supplier_join.finish()
        p_ps_join.finish()

        result = ps_l_shuffle(probe_pipeline=query20_probe_pipeline)
        agg_finalize_start = time.perf_counter_ns()
        query20_probe_pipeline.finalize()
        torch.cuda.synchronize(device=target_device)

        result["time_finalize_agg_ns"] = time.perf_counter_ns() - agg_finalize_start

        return result

    result_str = reader_wrapper.run(tmp_function)
    print(f"# supplier_join         : {supplier_join.get_stats()}")
    print(f"# p_ps_join             : {p_ps_join.get_stats()}")
    print(f"# ps_l_shuffle          : {ps_l_shuffle.get_stats()}")
    if iteration == 0:
        print(f"query,iteration,algo,{reader_wrapper.header_str}")
    print(f"query20,{iteration},{ps_l_shuffle.name()},{result_str}")
    ######################################################################################

    if os.environ["SHUFFLE_PROBE_TYPE"] != "FULL":
        return

    ######################################################################################
    groups, aggs = query20_probe_pipeline.get_results()
    torch.set_printoptions(linewidth=200,precision=4,sci_mode=False)
    print(f"# [Local] Result groups: {groups.shape=}")
    print("# suppkeys (first 5)         : ",groups[:5,0])
    print("# sum(l_quantity) (first 5)  : ",aggs[:5])

    # if os.environ.get("SYNC_RESULT", "False") == "True" and WORLD_SIZE > 1:
    #     print(f"# [Global] Results:")
    #     dist.all_reduce(query20_probe_pipeline, op=dist.ReduceOp.SUM)
    #     torch.cuda.synchronize(target_device)
    #     print("# revenue :", query20_probe_pipeline[0])

