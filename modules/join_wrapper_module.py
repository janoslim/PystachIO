import os
import math
import time
import torch
import torch.distributed as dist
import sys
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'helpers')))

from helpers.beemon import Beemon
from .morsel_table_module import MorselTableModule
from .aggregation_module import AggregationPostCombine

def join_wrapper(store, target_device, shuffle_module_type, left_data, right_data, left_index, right_index, morsel_mb, hash_init_num, iteration, probe_pipeline):
    """
        shuffle_module_type e.g. MorselShuffleWithJoin, MorselShuffleWithJoin_OtherGPU, MorselShuffleNetCompress
    """
    LOCAL_RANK = int(os.environ['LOCAL_RANK'])
    WORLD_SIZE = int(os.environ['WORLD_SIZE'])
    WORLD_RANK = int(os.environ['RANK'])
    SHUFFLE_RIGHT_REPEAT = int(os.environ['SHUFFLE_RIGHT_REPEAT'])

    if morsel_mb < 0:
        # no chunking:
        left_partition_row_count = left_data[0].shape[0]
        right_partition_row_count = right_data[0].shape[0]
    else:
        left_partition_row_count = math.floor(morsel_mb * (1<<20) / sum(col.dtype.itemsize for col in left_data))
        right_partition_row_count = math.floor(morsel_mb * (1<<20) / sum(col.dtype.itemsize for col in right_data))

    if iteration != -1:
        beemon = Beemon()

    left_table = MorselTableModule(left_data, left_partition_row_count)
    left_bytes = left_table.bytes()

    right_table = MorselTableModule(right_data, right_partition_row_count, full_repeat=SHUFFLE_RIGHT_REPEAT)
    right_bytes = right_table.bytes()

    transfer_size = left_table.tuples_per_morsel * left_table.columns[0].dtype.itemsize / WORLD_SIZE

    join_module = shuffle_module_type(target_device, hash_init_num, left_table, left_index,
                                      right_table, right_index, store, store_prefix=str(iteration))
    if iteration != -1:
        beemon.start(WORLD_RANK,num_repeat=iteration)
        time.sleep(0.2)

    torch.cuda.synchronize(device=target_device)
    dist.barrier()
    with torch.cuda.nvtx.range(f"MorselS {morsel_mb:.2f} MiB, TransferS {transfer_size/(1<<20):.2f} MiB"):
        results = join_module(probe_pipeline=probe_pipeline)

    start_agg_finalize = time.perf_counter_ns()
    if isinstance(probe_pipeline,AggregationPostCombine):
        probe_pipeline.finalize()

    torch.cuda.synchronize(device=target_device)
    time_finalize_agg_ns = time.perf_counter_ns()-start_agg_finalize 

    if iteration != -1:
        time.sleep(0.2)
        beemon.stop()

    time_part_ns = results.get("time_part_ns",-1)
    time_shuffle_ns =  results.get("time_shuffle_ns",-1)
    time_buildprobe_ns = results.get("time_buildprobe_ns",-1)

    total_bytes = torch.tensor([left_bytes+right_bytes],device="cpu")
    max_times = torch.tensor([time_part_ns, time_shuffle_ns, time_buildprobe_ns, time_finalize_agg_ns, results["time_ns"]],device="cpu")
    join_tuples_total = torch.tensor([results["local_result"]],device="cpu")

    dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)
    dist.all_reduce(max_times, op=dist.ReduceOp.MAX)
    dist.all_reduce(join_tuples_total, op=dist.ReduceOp.SUM)


    time_with_finalize_agg = max_times[3] + max_times[4]
    # how much data was sent out (same as received) of this node
    bus_bw_gibs = ((left_bytes+right_bytes)*((WORLD_SIZE-1)/WORLD_SIZE) / (1<<30)) / (time_with_finalize_agg / 1000000000)
    # this is the "global" average bud bandwidth, different from "local" above
    global_avg_bw_gibs = (total_bytes[0]*((WORLD_SIZE-1)/(WORLD_SIZE*WORLD_SIZE)) / (1<<30)) /(time_with_finalize_agg / 1000000000)

    env_var_info = ",".join(os.environ.get(key,"None") for key in ["NCCL_IB_DISABLE","NCCL_P2P_DISABLE","SHUFFLE_STREAMS","SHUFFLE_SEND_TYPE","SHUFFLE_RECV_TYPE","SHUFFLE_PROBE_TYPE","SHUFFLE_PROBE_SCHED","USE_RMM","USE_NET_OPT_FLAGS","SHUFFLE_AGG_GROUPS","AGG_USE_CONS_COUNT","SHUFFLE_ZIPF_LEFT","SHUFFLE_ZIPF_RIGHT"])

    return f"{join_module.name()},{left_bytes/(1<<30):.6f},{right_bytes/(1<<30):.6f},{WORLD_SIZE},{WORLD_RANK},{left_table.get_col_count()},{right_table.get_col_count()},{morsel_mb},{results['local_result']},{join_tuples_total[0]},{max_times[0]},{max_times[1]},{max_times[2]},{max_times[3]},{max_times[4]},{bus_bw_gibs:.6f},{global_avg_bw_gibs:.6f},{env_var_info}"

join_wrapper.header_str = "algo,left_table_size_gb,right_table_size_gb,nodes,node_id,cols_left,cols_right,morsel_mb,local_result,total_result,time_part_ns,time_shuffle_ns,time_buildprobe_ns,time_finalize_agg_ns,time_ns,bandwidth_gbs_single,bandwidth_gbs_global_avg,NCCL_IB_DISABLE,NCCL_P2P_DISABLE,SHUFFLE_STREAMS,SHUFFLE_SEND_TYPE,SHUFFLE_RECV_TYPE,SHUFFLE_PROBE_TYPE,SHUFFLE_PROBE_SCHED,USE_RMM,USE_NET_OPT_FLAGS,SHUFFLE_AGG_GROUPS,AGG_USE_CONS_COUNT,SHUFFLE_ZIPF_LEFT,SHUFFLE_ZIPF_RIGHT"