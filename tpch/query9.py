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
from modules.tensor_buffer import TensorBuffer
from modules.query_wrapper import ReaderWrapper
from modules.single_node_join import SingleNodeJoin
from modules.aggregation_module import Aggregation,AggregationPostCombine
from .tpch import get_orders,get_customer,get_lineitem
from .dict import *

QUERY9_SQL = """
select
        s_nationkey as nation,
        extract(year from o_orderdate) as o_year,
        sum(l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity) as amount
from
        part,
        supplier,
        lineitem,
        partsupp,
        orders
where
        s_suppkey = l_suppkey
        and ps_suppkey = l_suppkey
        and ps_partkey = l_partkey
        and p_partkey = l_partkey
        and o_orderkey = l_orderkey
        and p_size <= 3
group by
        s_nationkey,
        o_year
order by
        s_nationkey,
        o_year

"""

def query9_agg(sorted_data,groups_unique,group_idx):
    # s_nationkey, year, ps_supplycost, l_quantity, l_extendedprice, l_discount
    # l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity
    amount = sorted_data[4] * (100 - sorted_data[5]) - sorted_data[2] * sorted_data[3]

    results = torch.zeros(groups_unique.shape[0],dtype=amount.dtype,device=sorted_data[0].get_device()).scatter_reduce(0, group_idx, amount, reduce="sum")
    return results

def query9_agg_combine(sorted_data,groups_unique,group_idx):
    results = torch.zeros(groups_unique.shape[0],dtype=sorted_data[1].dtype,device=sorted_data[0].get_device()).scatter_reduce(0, group_idx, sorted_data[1], reduce="sum")
    return results


def query9_probe_pipeline(aggregation, joined_tuples):
    # after shuffle:
    # ["s_suppkey", "s_nationkey","p_partkey", "p_size","ps_partkey","ps_suppkey","ps_supplycost","l_orderkey",
    #  "l_partkey","l_suppkey","l_quantity","l_extendedprice","l_discount","o_orderkey", "o_orderdate"]

    # not exactly year, but close enough
    # year = torch.div(joined_tuples[10], 365, rounding_mode="floor") + 1970
    # aggregation([joined_tuples[5], joined_tuples[1], year, joined_tuples[8], joined_tuples[9]])
    joined_tuples[14] = torch.div(joined_tuples[14], 365, rounding_mode="floor") + 1970
    # s_nationkey, year, ps_supplycost, l_quantity, l_extendedprice, l_discount
    aggregation([joined_tuples[1],joined_tuples[14],joined_tuples[6],joined_tuples[10],joined_tuples[11],joined_tuples[12]])


def run_query9_on_parquet(iteration, store, target_device, scale_factor):
    WORLD_SIZE = int(os.environ['WORLD_SIZE'])
    if os.environ["SHUFFLE_TYPE"] == "BLOCKING":
        SHUFFLE_MODULE_TYPE = ShuffleWithJoinNaive
    elif "OVERLAPPED" in os.environ["SHUFFLE_TYPE"]:
        SHUFFLE_MODULE_TYPE = ShuffleWithJoin_MultiStream
    else:
        print(f"Error, unknown SHUFFLE_TYPE={os.environ["SHUFFLE_TYPE"]}")
        return
    storage_threads = int(os.environ["STORAGE_THREADS"])

    base_folder = f"{os.environ['STORAGE_BASE_PATH']}/tpch-sf{scale_factor}/"

    reader_wrapper = ReaderWrapper(store, target_device,
                                         # name, filepath, threads, cols, filters
                                table_infos= [  
                                                ("supplier", f"{base_folder}/supplier.parquet", storage_threads,
                                                    ["s_suppkey", "s_nationkey"],None
                                                ),

                                                ("part", f"{base_folder}/part.parquet", storage_threads,
                                                    ["p_partkey", "p_size"],
                                                    TableFilter({
                                                        1: [FilterCondition(FilterOp.LE, 3)],     # p_size <= 3
                                                    }),
                                                ),

                                                ("partsupp", f"{base_folder}/partsupp.parquet", storage_threads,
                                                    ["ps_partkey","ps_suppkey","ps_supplycost"],
                                                    None
                                                ),

                                                ("lineitem", f"{base_folder}/lineitem.parquet", storage_threads,
                                                    ["l_orderkey","l_partkey","l_suppkey","l_quantity","l_extendedprice","l_discount"],
                                                    None
                                                ),

                                                ("orders", f"{base_folder}/orders.parquet", storage_threads,
                                                    ["o_orderkey", "o_orderdate"], None
                                                ),

                                        ])

    supplier_reader, part_reader, partsupp_reader, lineitem_reader, orders_reader = reader_wrapper.readers
    if os.environ["HASH_SIZE_NAIVE"] == "True":
        p_ps_hash_init = (16000*scale_factor)
        pps_l_hash_init = (120000*scale_factor)
        s_ppsl_hash_init = (TPCH_CATALOG["supplier"]["size"](scale_factor)+1000)
        shuffle_hash_init =  (400000*scale_factor) // WORLD_SIZE
    else:
        p_ps_hash_init = util.next_prime(16000*scale_factor)
        pps_l_hash_init = util.next_prime(120000*scale_factor)
        s_ppsl_hash_init = util.next_prime((TPCH_CATALOG["supplier"]["size"](scale_factor)+1000))
        shuffle_hash_init =  util.next_prime((400000*scale_factor) // WORLD_SIZE)

    torch.cuda.empty_cache()
    import gc
    gc.collect()

    p_ps_join = SingleNodeJoin(target_device, p_ps_hash_init, part_reader.get_fifo(),
                                            right_child=partsupp_reader.get_fifo(), right_col_idx=0)

    pps_l_join = SingleNodeJoin(target_device, pps_l_hash_init, p_ps_join,
                                            right_child=lineitem_reader.get_fifo(), right_col_idx=1,    # join l_partkey
                                            add_join_attrib=[(3,2)])                                    # join suppkey

    s_ppsl_join = SingleNodeJoin(target_device, s_ppsl_hash_init, supplier_reader.get_fifo(),
                                            right_child=pps_l_join, right_col_idx=3)

    shuffle_left_child = s_ppsl_join
    shuffle_right_child = orders_reader.get_fifo()

    if (tb_size := os.environ.get("SHUFFLE_BUFFER_LEFT")) is not None:
        print(f"# Adding tensor buffer for left join side (size {int(tb_size)})")
        shuffle_left_child = TensorBuffer(s_ppsl_join, int(tb_size))
    if (tb_size := os.environ.get("SHUFFLE_BUFFER_RIGHT")) is not None:
        print(f"# Adding tensor buffer for right join side (size {int(tb_size)})")
        shuffle_right_child = TensorBuffer(orders_reader.get_fifo(), int(tb_size))

    # build side: ["s_suppkey", "s_nationkey","p_partkey", "p_size","ps_partkey","ps_suppkey","ps_supplycost","l_orderkey",
    #               "l_partkey","l_suppkey","l_quantity","l_extendedprice","l_discount"],
    # probe side: ["o_orderkey", "o_orderdate"]
    shuffle_join = SHUFFLE_MODULE_TYPE(target_device, shuffle_hash_init, shuffle_left_child, 7,
                                         shuffle_right_child, 0, store, store_prefix="")

    # projection done in query9_probe_pipeline
    query9_aggregation = AggregationPostCombine(query9_agg, query9_agg_combine, [0,1])
    probe_pipeline = partial(query9_probe_pipeline, query9_aggregation)

    ######################################################################################
    def tmp_function():
        build_start = time.perf_counter_ns()
        s_ppsl_join.build(supplier_reader.get_fifo(), 0)                  # build on s_suppkey
        p_ps_join.build(part_reader.get_fifo(), 0)     # build on p_partkey

        p_ps_join.finish()
        pps_l_join.build(p_ps_join, 0)                  # build on p_partkey
        s_ppsl_join.finish()
        pps_l_join.finish()

        build_end = time.perf_counter_ns()
        # print(f"Single Node build sides took {(build_end-build_start)/1e6:.6f}ms")

        result = shuffle_join(probe_pipeline=probe_pipeline)
        agg_finalize_start = time.perf_counter_ns()
        query9_aggregation.finalize()
        torch.cuda.synchronize(device=target_device)

        result["time_finalize_agg_ns"] = time.perf_counter_ns() - agg_finalize_start
        return result

    result_str = reader_wrapper.run(tmp_function)

    print(f"# {p_ps_join.get_stats()=}")
    print(f"# {pps_l_join.get_stats()=}")
    print(f"# {s_ppsl_join.get_stats()=}")
    print(f"# {shuffle_join.get_stats()=}")

    if iteration == 0:
        print(f"query,iteration,algo,{reader_wrapper.header_str}")
    print(f"query9,{iteration},{shuffle_join.name()},{result_str}")
    ######################################################################################

    if os.environ["SHUFFLE_PROBE_TYPE"] != "FULL":
        return

    ######################################################################################
    groups, aggs = query9_aggregation.get_results()

    torch.set_printoptions(linewidth=200,precision=4,sci_mode=False)
    print(f"# [Local] Result groups: {groups.shape=}")
    print("# s_nationkey    : ",groups[:10,0])
    print("# year           : ",groups[:10,1])
    print("# amount         : ",aggs[:10])

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
                print("# s_nationkey : ",unique_groups[:10,0])
                print("# year        : ",unique_groups[:10,1])
                print("# amount      : ",final_aggs[:10])

    torch.set_printoptions()