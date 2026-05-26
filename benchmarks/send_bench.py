import sys
import os
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
from datetime import timedelta
import time

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroupNCCL
import faulthandler

import rmm
from rmm.allocators.torch import rmm_torch_allocator

import modules.send_module
from helpers.nccl_init import nccl_init, nccl_init_sync
from modules.partitioning_module import ModuloSharding


def simple_morsel_all_to_all(target_device, world_rank, world_size, morsel_mbs, comm="all_to_all"):
    print("algo,nodes,node_id,morsel_mb,rounds,moved_gb,time_ns,bandwidth_gbs_single,bandwidth_gbs_total,USE_RMM,USE_NET_OPT_FLAGS")

    stream = torch.cuda.Stream(device=target_device)
    for iteration,morsel_mb in enumerate(morsel_mbs):
        morsel_size = int(morsel_mb * (1<<20))
        # prepare a tensor to be send out to each other node
        if comm == "all_to_all":
            send_tensors = [torch.arange(morsel_size, dtype=torch.int8, device=target_device) for _ in range(world_size)]
            recv_tensor = [torch.empty((morsel_size,), dtype=torch.int8, device=target_device) for _ in range(world_size)]
        elif comm == "all_to_all_single":
            send_tensors = torch.empty(world_size*morsel_size, dtype=torch.int8, device=target_device)
            recv_tensor = torch.empty((world_size*morsel_size,), dtype=torch.int8, device=target_device)
            # [morsel_size, morsel_size, morsel_size,morsel_size]
            input_splits = [morsel_size]*world_size
            output_splits = input_splits.copy()

        gb_per_round = (world_size*morsel_mb) / (1<<10)

        rounds = max(10, int(4 / gb_per_round))

        torch.cuda.synchronize(target_device)
        dist.barrier()

        start = time.perf_counter_ns()
        # start of timed part
        with torch.cuda.stream(stream):
            torch.cuda.nvtx.range_push(f"SimpleMorsel AllToAll ({morsel_mb=})")
            for _ in range(rounds):
                if comm == "all_to_all":
                    dist.all_to_all(input_tensor_list=send_tensors, output_tensor_list=recv_tensor)
                elif comm == "all_to_all_single":
                    # dist.all_to_all_single(input_tensor_list=send_tensors, output_tensor_list=recv_tensor)
                    dist.all_to_all_single(input=send_tensors, output=recv_tensor, input_split_sizes=input_splits, output_split_sizes=output_splits)
                else:
                    print(f"Unknown {comm=}")
        torch.cuda.synchronize(target_device)
        end = time.perf_counter_ns()
        # end of timed part
        torch.cuda.nvtx.range_pop()

        time_ns = end-start
        max_time_ns = torch.tensor([time_ns],device=target_device)
        dist.all_reduce(max_time_ns, op=dist.ReduceOp.MAX)
        torch.cuda.synchronize(target_device)

        # print(f"{gb_per_round=}, {rounds=}, {(gb_per_round*rounds)=}, {time_ns / 1000000000}")
        bw_gibs_total = (gb_per_round*rounds) / (time_ns / 1000000000)
        bw_gibs_single = bw_gibs_total * (world_size-1) / world_size
        USE_RMM = os.environ.get("USE_RMM", "False")
        USE_NET_OPT_FLAGS = os.environ.get("USE_NET_OPT_FLAGS", "False")
        print(f"const_{comm},{world_size},{world_rank},{morsel_mb},{rounds},{gb_per_round*rounds},{time_ns},{bw_gibs_single},{bw_gibs_total},{USE_RMM},{USE_NET_OPT_FLAGS}")


def comm_sizes_all_to_all(store, target_device, world_rank, world_size, morsel_mbs, num_cols, comm="store"):
    print("algo,nodes,node_id,num_cols,morsel_mb,rounds,moved_gb,time_ns,bandwidth_gbs_single,bandwidth_gbs_total,USE_RMM,USE_NET_OPT_FLAGS")

    stream = torch.cuda.Stream(device=target_device)

    for iteration,morsel_mb in enumerate(morsel_mbs):
        tuple_per_morsel = int((morsel_mb * (1<<20)) / (num_cols*8))
        # print(f"{tuple_per_morsel=}")
        # prepare a tensor to be send out to each other node
        if comm != "inplace_ccl":
            send_tensors = [[torch.arange(tuple_per_morsel, dtype=torch.int64, device=target_device) for _ in range(num_cols)] for _ in range(world_size)]
        else:
            # comm==inplace_ccl
            send_tensors = [torch.arange(world_size*tuple_per_morsel, dtype=torch.int64, device=target_device) for _ in range(num_cols)]
            sharder = ModuloSharding(world_size)
            partitioned,input_splits = sharder.shard_in_place(send_tensors, 0, target_device)
            # print("send_tensors:",send_tensors)
            # print("partitioned:",partitioned[0].shape,partitioned)
            # print("input_splits:",input_splits)
            # input_splits = torch.zeros(world_size, dtype=torch.int64, device=target_device)
            # input_splits[(world_rank+1) % world_size] = send_tensors[0].shape[0]

        # recv_tensor = [torch.empty((tuple_per_morsel,), dtype=torch.int64, device=target_device) for _ in range(world_size)]

        gb_per_round = (world_size*morsel_mb) / (1<<10)

        rounds = max(10, int(4 / gb_per_round))

        transfer_size = send_tensors[0][0].numel()*send_tensors[0][0].dtype.itemsize

        torch.cuda.synchronize(target_device)
        dist.barrier()

        start = time.perf_counter_ns()
        with torch.cuda.stream(stream):
            torch.cuda.nvtx.range_push(f"CommSizes AllToAll ({morsel_mb=}, {transfer_size/ (1<<20):.3f} MiB Transfers, {comm=})")
            # start of timed part
            for r in range(rounds):
                if comm == "store":
                    recv = modules.send_module.send_with_store(store, f"{iteration}_{r}", world_rank, world_size, iteration, send_tensors,
                                                        [torch.int64]*num_cols, target_device=target_device)
                elif comm == "ccl":
                    recv = modules.send_module.send_with_ccl(world_size, send_tensors,
                                                              [torch.int64]*num_cols, target_device=target_device)
                elif comm == "ccl_combine_recv":
                    recv = modules.send_module.send_with_ccl_combine_recv(world_size, send_tensors,
                                                              [torch.int64]*num_cols, target_device=target_device)
                elif comm == "inplace_ccl":
                    recv = modules.send_module.send_inplace_with_ccl(world_size, partitioned, input_splits,
                                                         [torch.int64]*num_cols, target_device=target_device)
        torch.cuda.synchronize(target_device)
        end = time.perf_counter_ns()
        # end of timed part
        torch.cuda.nvtx.range_pop()

        time_ns = end-start
        max_time_ns = torch.tensor([time_ns],device=target_device)
        dist.all_reduce(max_time_ns, op=dist.ReduceOp.MAX)
        torch.cuda.synchronize(target_device)
        bw_gibs_total = (gb_per_round*rounds) / (time_ns / 1000000000)
        bw_gibs_single = bw_gibs_total * (world_size-1) / world_size

        USE_RMM = os.environ.get("USE_RMM", "False")
        USE_NET_OPT_FLAGS = os.environ.get("USE_NET_OPT_FLAGS", "False")

        print(f"{comm},{world_size},{world_rank},{num_cols},{morsel_mb},{rounds},{gb_per_round*rounds},{time_ns},{bw_gibs_single},{bw_gibs_total},{USE_RMM},{USE_NET_OPT_FLAGS}")


def const_size_all_to_all_with_compute(store, target_device, world_rank, world_size, morsel_mbs, comm="all_to_all"):
    print("algo,nodes,node_id,morsel_mb,rounds,moved_gb,time_ns,bandwidth_gbs_single,bandwidth_gbs_total,USE_RMM,USE_NET_OPT_FLAGS")

    compute_data = torch.arange((1<<26), dtype=torch.int64, device=target_device)
    shuffle_stream = torch.cuda.Stream(device=target_device)
    compute_stream = torch.cuda.Stream(device=target_device)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for iteration,morsel_mb in enumerate(morsel_mbs):
        morsel_size = int(morsel_mb * (1<<20))
        # prepare a tensor to be send out to each other node
        if comm == "all_to_all":
            send_tensors = [torch.arange(morsel_size, dtype=torch.int8, device=target_device) for _ in range(world_size)]
            recv_tensor = [torch.empty((morsel_size,), dtype=torch.int8, device=target_device) for _ in range(world_size)]
        elif comm == "all_to_all_single":
            send_tensors = torch.empty(world_size*morsel_size, dtype=torch.int8, device=target_device)
            recv_tensor = torch.empty((world_size*morsel_size,), dtype=torch.int8, device=target_device)
            # [morsel_size, morsel_size, morsel_size,morsel_size]
            input_splits = [morsel_size]*world_size
            output_splits = input_splits.copy()

        gb_per_round = (world_size*morsel_mb) / (1<<10)

        rounds = max(10, int(4 / gb_per_round))

        torch.cuda.synchronize(target_device)
        dist.barrier()

        # do some computation:
        with torch.cuda.stream(compute_stream):
            for i in range(1000):
                compute_data += 1
                compute_data -= 1

        # shuffle
        with torch.cuda.stream(shuffle_stream):
            torch.cuda.nvtx.range_push(f"AllToAll With Compute ({morsel_mb=})")
            start_event.record(shuffle_stream)
            for _ in range(rounds):
                if comm == "all_to_all":
                    dist.all_to_all(input_tensor_list=send_tensors, output_tensor_list=recv_tensor)
                elif comm == "all_to_all_single":
                    # dist.all_to_all_single(input_tensor_list=send_tensors, output_tensor_list=recv_tensor)
                    dist.all_to_all_single(input=send_tensors, output=recv_tensor, input_split_sizes=input_splits, output_split_sizes=output_splits)
                else:
                    print(f"Unknown {comm=}")
            end_event.record(shuffle_stream)

        torch.cuda.synchronize(target_device)

        torch.cuda.nvtx.range_pop()

        time_ns = start_event.elapsed_time(end_event) * 1e6
        max_time_ns = torch.tensor([time_ns],device=target_device)
        dist.all_reduce(max_time_ns, op=dist.ReduceOp.MAX)
        torch.cuda.synchronize(target_device)
        bw_gibs_total = (gb_per_round*rounds) / (time_ns / 1000000000)
        bw_gibs_single = bw_gibs_total * (world_size-1) / world_size
        USE_RMM = os.environ.get("USE_RMM", "False")
        USE_NET_OPT_FLAGS = os.environ.get("USE_NET_OPT_FLAGS", "False")
        print(f"const_{comm}_compute,{world_size},{world_rank},{morsel_mb},{rounds},{gb_per_round*rounds},{time_ns},{bw_gibs_single},{bw_gibs_total},{USE_RMM},{USE_NET_OPT_FLAGS}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <device_id>")
        exit(1)

    if os.environ.get("USE_RMM", "False") == "True":
        mr = rmm.mr.CudaAsyncMemoryResource()
        rmm.mr.set_current_device_resource(mr)
        torch.cuda.memory.change_current_allocator(rmm_torch_allocator)

    os.environ.setdefault("NUM_COLS","8")
    os.environ.setdefault("MORSEL_MB_LO","0.25")
    os.environ.setdefault("MORSEL_MB_HI","4096")
    os.environ.setdefault("SEND_BENCH","COMM_SIZE") # [CONST_SEND,COMM_SIZE]

    LOCAL_RANK = int(os.environ['LOCAL_RANK'])
    WORLD_SIZE = int(os.environ['WORLD_SIZE'])
    WORLD_RANK = int(os.environ['RANK'])
    MASTER_IP = os.environ['MASTER_ADDR']
    TCP_STORE_PORT = int(os.environ['TCP_STORE_PORT'])
    target_device = int(sys.argv[1])

    num_cols = int(os.environ["NUM_COLS"])
    repeats = int(os.environ["NUM_REPEATS"])
    SEND_BENCH = os.environ["SEND_BENCH"]

    # print("Visible devices:", os.environ["CUDA_VISIBLE_DEVICES"])

    store = dist.TCPStore(MASTER_IP, TCP_STORE_PORT, WORLD_SIZE, WORLD_RANK == 0, timedelta(seconds=30))
    dist.init_process_group("cuda:nccl,cpu:gloo", store=store, rank=WORLD_RANK, world_size=WORLD_SIZE, device_id=torch.device(target_device))

    faulthandler.enable()
    nccl_init(target_device)
    nccl_init_sync(target_device)
    torch.cuda.synchronize(target_device)

    morsel_mbs = [float(os.environ["MORSEL_MB_LO"])]
    while morsel_mbs[-1] < float(os.environ["MORSEL_MB_HI"]):
        morsel_mbs.append(morsel_mbs[-1]*2)
    morsel_mbs *= repeats

    with torch.jit.optimized_execution(False):
        with torch.inference_mode():
            if SEND_BENCH == "CONST_SEND":
                simple_morsel_all_to_all(target_device, WORLD_RANK, WORLD_SIZE, morsel_mbs, comm="all_to_all")
                # simple_morsel_all_to_all(target_device, WORLD_RANK, WORLD_SIZE, morsel_mbs, comm="all_to_all_single")

            elif SEND_BENCH == "COMM_SIZE":
                # comm_sizes_all_to_all(store, target_device, WORLD_RANK, WORLD_SIZE, morsel_mbs, num_cols=num_cols, comm="store")
                comm_sizes_all_to_all(store, target_device, WORLD_RANK, WORLD_SIZE, morsel_mbs, num_cols=num_cols, comm="ccl")
                # comm_sizes_all_to_all(store, target_device, WORLD_RANK, WORLD_SIZE, morsel_mbs, num_cols=num_cols, comm="ccl_combine_recv")
                # comm_sizes_all_to_all(store, target_device, WORLD_RANK, WORLD_SIZE, morsel_mbs, num_cols=num_cols, comm="inplace_ccl")
            else:
                print("WARN: Unknown SEND_BENCH value:",SEND_BENCH)
            # with computation:
            # const_size_all_to_all_with_compute(store, target_device, WORLD_RANK, WORLD_SIZE, morsel_mbs, comm="all_to_all")
    dist.destroy_process_group()