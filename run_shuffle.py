#!/usr/bin/env python3
import faulthandler
import math
import os
import time
import gc
import sys
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroupNCCL

import rmm
from rmm.allocators.torch import rmm_torch_allocator

from helpers.beemon import Beemon
from helpers.util import convert_size,data_creation,next_prime
from helpers.nccl_init import nccl_init, nccl_init_sync

from modules.join_wrapper_module import join_wrapper
from modules.morsel_table_module import MorselTableModule

from modules.single_thread_naive import ShuffleWithJoinNaive
from modules.single_thread_shuffle_join_multistream import ShuffleWithJoin_MultiStream
from modules.single_thread_shuffle_mig import ShuffleWithJoin_MIG

from modules.aggregation_module import AggregationPostCombine

##########################################################################################
class DummyStore:
    def __init__(self,host):
        self.host = host

def query_agg_scatter(sorted_data,groups_unique,group_idx):
    results = torch.zeros(groups_unique.shape[0],dtype=sorted_data[0].dtype,device=sorted_data[0].get_device()).scatter_reduce(0, group_idx, sorted_data[0], reduce=reduce)
    return results

def query_agg_counts(sorted_data,groups_unique,group_counts):
    num_groups = groups_unique.shape[0]
    results = torch.zeros(num_groups,dtype=sorted_data[0].dtype,device=sorted_data[0].get_device())
    prefix_sum = torch.zeros(num_groups+1, dtype=torch.int64, device="cuda:0")
    sums = torch.zeros(num_groups,dtype=torch.int64,device=sorted_data[0].get_device())
    torch.cumsum(group_counts, dim=0, out=prefix_sum[1:])
    for i in range(num_groups):
        results[i] = sorted_data[0][prefix_sum[i]:prefix_sum[i+1]].sum()

    return results

def query_agg_combine_scatter(sorted_data,groups_unique,group_idx):
    results = torch.zeros(groups_unique.shape[0],dtype=sorted_data[1].dtype,device=sorted_data[0].get_device()).scatter_reduce(0, group_idx, sorted_data[1], reduce="sum")
    return results

def query_agg_combine_counts(sorted_data,groups_unique,group_counts):
    num_groups = groups_unique.shape[0]
    results = torch.zeros(num_groups,dtype=sorted_data[0].dtype,device=sorted_data[0].get_device())
    prefix_sum = torch.zeros(num_groups+1, dtype=torch.int64, device="cuda:0")
    sums = torch.zeros(num_groups,dtype=torch.int64,device=sorted_data[0].get_device())
    torch.cumsum(group_counts, dim=0, out=prefix_sum[1:])
    for i in range(num_groups):
        results[i] = sorted_data[1][prefix_sum[i]:prefix_sum[i+1]].sum()
    return results

##########################################################################################

def run_in_memory_shuffle(world_rank, world_size, store, target_device, num_cols, morsel_mbs, repeats):
    if (SHUFFLE_LEFT_TUPLES := os.environ.get("SHUFFLE_LEFT_TUPLES")) is not None:
        left_target  = int(SHUFFLE_LEFT_TUPLES)
    else:
        left_target  =  int(200000000 // num_cols)
    if (SHUFFLE_RIGHT_TUPLES := os.environ.get("SHUFFLE_RIGHT_TUPLES")) is not None:
        right_target = int(SHUFFLE_RIGHT_TUPLES)
    else:
        right_target = int(400000000 // num_cols)

    SHUFFLE_JOIN_RATIO = float(os.environ.get("SHUFFLE_JOIN_RATIO", "1.0"))
    SHUFFLE_ZIPF_LEFT = float(os.environ.get("SHUFFLE_ZIPF_LEFT", "0.0"))
    SHUFFLE_ZIPF_RIGHT = float(os.environ.get("SHUFFLE_ZIPF_RIGHT", "0.0"))

    left_data = data_creation(target_device, num_cols, world_rank, world_size, left_target*world_size, SHUFFLE_ZIPF_LEFT)
    right_data = data_creation(target_device, num_cols, world_rank, world_size, right_target*world_size, SHUFFLE_ZIPF_RIGHT)

    remainder = int((left_target*world_size) / SHUFFLE_JOIN_RATIO)
    if int((right_target*world_size)) < remainder:
        print(f"# WARN: SHUFFLE_JOIN_RATIO not enforceable this way!")
    right_data[0].remainder_(remainder)

    hash_init_num = next_prime(int(1.2*left_target))

    probe_pipeline = lambda *args, **kwargs: None
    AGG_USE_CONS_COUNT = bool(os.environ.get("AGG_USE_CONS_COUNT", "False"))

    if (SHUFFLE_AGG_GROUPS := os.environ.get("SHUFFLE_AGG_GROUPS")) is not None:
        assert num_cols > 1
        left_data[1].remainder_(int(SHUFFLE_AGG_GROUPS))

        # (this aggregation is just set here for the warmup)
        if AGG_USE_CONS_COUNT:
            probe_pipeline = AggregationPostCombine(query_agg_counts, query_agg_combine_counts, [1], projection=[1], use_consecutive_counts=AGG_USE_CONS_COUNT)
        else:
            probe_pipeline = AggregationPostCombine(query_agg_scatter, query_agg_combine_scatter, [1], projection=[1], use_consecutive_counts=AGG_USE_CONS_COUNT)

    if os.environ["SHUFFLE_TYPE"] == "BLOCKING":
        SHUFFLE_MODULE_TYPE = ShuffleWithJoinNaive
    elif os.environ["SHUFFLE_TYPE"] == "MIG":
        SHUFFLE_MODULE_TYPE = ShuffleWithJoin_MIG
    elif "OVERLAPPED" in os.environ["SHUFFLE_TYPE"]:
        SHUFFLE_MODULE_TYPE = ShuffleWithJoin_MultiStream
    else:
        print(f"Error, unknown SHUFFLE_TYPE={os.environ["SHUFFLE_TYPE"]}")
        return

    # warm up
    join_wrapper(store, target_device, SHUFFLE_MODULE_TYPE, left_data, right_data, 0, 0, 512,
                     hash_init_num=hash_init_num,
                     iteration=-1, probe_pipeline=probe_pipeline)
    torch.cuda.empty_cache()
    gc.collect()

    morsel_mbs = morsel_mbs*repeats

    print(join_wrapper.header_str)
    for iteration,morsel_mb in enumerate(morsel_mbs):
        if (SHUFFLE_AGG_GROUPS := os.environ.get("SHUFFLE_AGG_GROUPS")) is not None:
            if AGG_USE_CONS_COUNT:
                probe_pipeline = AggregationPostCombine(query_agg_counts, query_agg_combine_counts, [1], projection=[1], use_consecutive_counts=AGG_USE_CONS_COUNT)
            else:
                probe_pipeline = AggregationPostCombine(query_agg_scatter, query_agg_combine_scatter, [1], projection=[1], use_consecutive_counts=AGG_USE_CONS_COUNT)
        res_str = join_wrapper(store, target_device, SHUFFLE_MODULE_TYPE, left_data, right_data, 0, 0, morsel_mb,
                     hash_init_num=hash_init_num,
                     iteration=iteration, probe_pipeline=probe_pipeline)
        torch.cuda.empty_cache()
        gc.collect()
        # print(rmm.statistics.get_statistics())
        print(res_str)
        if (SHUFFLE_AGG_GROUPS := os.environ.get("SHUFFLE_AGG_GROUPS")) is not None:
            groups, aggs = probe_pipeline.get_results()
            print(f"# [Local] Result groups: {groups.shape=}")
            print("# group keys (first 5)    : ",groups[:5,0])
            print("# groups aggs (first 5)   : ",aggs[:5])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <target_device_id>")
        exit(1)

    if os.environ.get("USE_RMM", "False") == "True":
        mr = rmm.mr.CudaAsyncMemoryResource()
        rmm.mr.set_current_device_resource(mr)
        torch.cuda.memory.change_current_allocator(rmm_torch_allocator)
        # rmm.statistics.enable_statistics()

    os.environ.setdefault("NUM_COLS","6")
    os.environ.setdefault("MORSEL_MB_LO","128")
    os.environ.setdefault("MORSEL_MB_HI","2048")

    WORLD_SIZE = int(os.environ['WORLD_SIZE'])
    WORLD_RANK = int(os.environ['RANK'])
    MASTER_IP = os.environ['MASTER_ADDR']
    TCP_STORE_PORT = int(os.environ['TCP_STORE_PORT'])
    target_device = int(sys.argv[1])
    num_cols = int(os.environ["NUM_COLS"])
    repeats = int(os.environ["NUM_REPEATS"])
    morsel_mbs = [int(os.environ["MORSEL_MB_LO"])]
    if morsel_mbs[-1] > 0:
        while morsel_mbs[-1] < int(os.environ["MORSEL_MB_HI"]):
            morsel_mbs.append(morsel_mbs[-1]*2)

    # store = DummyStore(MASTER_IP)
    store = dist.TCPStore(MASTER_IP, TCP_STORE_PORT, WORLD_SIZE, WORLD_RANK == 0, timedelta(seconds=30))
    dist.init_process_group("cuda:nccl,cpu:gloo", store=store, rank=WORLD_RANK, world_size=WORLD_SIZE, device_id=torch.device(target_device))

    faulthandler.enable()
    if WORLD_SIZE>1:
        nccl_init(target_device)
        nccl_init_sync(target_device)
        torch.cuda.synchronize(target_device)

    # print(f"Using GPU: {os.environ.get("CUDA_VISIBLE_DEVICES")}, {target_device}")
    with torch.jit.optimized_execution(False):
        with torch.inference_mode():
            run_in_memory_shuffle(WORLD_RANK, WORLD_SIZE, store, target_device, num_cols, morsel_mbs, repeats)
    dist.destroy_process_group()
