import sys
import os
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'modules')))
from partitioning_module import ModuloSharding
from morsel_table_module import MorselTableModule
import torch
import time
import math

import rmm
from rmm.allocators.torch import rmm_torch_allocator

def test(device_id, partitions, cols, morsel_mbs):
    device = torch.device(device_id)
    sharder = ModuloSharding(partitions)
    # data = [torch.arange(0, 40, 1, dtype=torch.int64,device=device) for _ in range(cols)]
    data = [torch.arange(0, (1<<30), 1, dtype=torch.int64,device=device) for _ in range(cols)]

    data_size_gb = sum([x.element_size() * x.nelement() for x in data]) / (1<<30)
    # print(f"Data size: {data_size_gb:5.6f} GB")
    print("algo,partitions,cols,data_size_gb,morsel_mb,time_us,proc_bw_gibs")

    for idx,fun in enumerate([
                             sharder.shard_list_dynamic,sharder.shard_list_dynamic_mask,
                sharder.shard_list_dynamic_where,
                sharder.shard_in_place,
                sharder.shard_list_really_nosync
                ]):

        for morsel_mb in morsel_mbs:
            row_count = math.floor(morsel_mb * (1<<20) / sum(col.dtype.itemsize for col in data))
            table = MorselTableModule(data, row_count)
            # dummy call for initalization
            _ = fun([torch.arange(0, (1<<20), 1, dtype=torch.int64,device=device) for _ in range(1)], 0, device)
            torch.cuda.synchronize(device)

            # extra stream makes the overview clearer
            stream = torch.cuda.Stream(device=device)
            with torch.cuda.stream(stream):
                torch.cuda.nvtx.range_push(f"{idx} {fun.__name__} {morsel_mb}")
                # torch.cuda.set_sync_debug_mode(1)
                start = time.perf_counter_ns()
                while (next_chunk := table()) is not None:
                    # torch.cuda.set_sync_debug_mode(1)
                    _ = fun(next_chunk, 0, device)
                    # torch.cuda.set_sync_debug_mode(0)
                    # print("Partitioned:",_)
                    # if fun == sharder.shard_list_really_nosync:
                        # the_data,the_sizes = _
                        # partitioned_data = [[col[:size] for col in next_chunk] for next_chunk,size in zip(the_data,the_sizes)]
                        # print(partitioned_data)
                    # else:
                        # print(_)
                # torch.cuda.set_sync_debug_mode(0)
                stream.synchronize()
                torch.cuda.nvtx.range_pop()
                end = time.perf_counter_ns()
                # print("---------------------")
                # print(_)
                time_us = (end-start) / 1000
                proc_bw_gibs = data_size_gb / (time_us / 1000000)
                # print(f"{fun.__name__} took {time_us:10.2f} us, BW: {proc_bw_gibs:5.6f}GiB/s")
                print(f"{fun.__name__},{partitions},{cols},{data_size_gb},{morsel_mb:.6f},{time_us:.6f},{proc_bw_gibs:.6f}")


def test_unsync(device_id):
    # >>> a = Torch.tensor([[ 1,  2,  3,  4,  5,  6], [10, 20, 30, 40, 50, 60]])
    # >>> selectors = Torch.tensor([[ True, False, False,  True, False,  True], [False, False,  True,  True, False, False]])
    # >>> torch.cumsum(selectors.int(), dim=1)
    # tensor([[1, 1, 1, 2, 2, 3],
    #         [0, 0, 1, 2, 2, 2]])
    # >>> indices = torch.masked_fill(torch.cumsum(selectors.int(), dim=1), ~selectors, 0)
    # >>> indices
    # tensor([[1, 0, 0, 2, 0, 3],
    #         [0, 0, 1, 2, 0, 0]])
    # >>> torch.scatter(input=torch.zeros_like(a), dim=1, index=indices, src=a)
    # tensor([[ 5,  1,  4,  6,  0,  0],
    #         [60, 30, 40,  0,  0,  0]])
    # >>> torch.scatter(input=torch.zeros_like(a), dim=1, index=indices, src=a)[:,1:]
    # tensor([[ 1,  4,  6,  0,  0],
    #         [30, 40,  0,  0,  0]])
    device = torch.device(device_id)
    # torch.cuda.set_sync_debug_mode(1)
    a = torch.arange(0, 20, 1, dtype=torch.int64,device=device)
    print("a:",a)
    mod = a.remainder(2)
    # mask = torch.zeros_like(mod, dtype=torch.bool, device=device)
    # mask[mod == 1] = True
    # mask = torch.tensor([True, False, False,  True, False,  True, True, False, False,  True, False,  True,True, False, False,  True, False,  True,False,False],device=device)
    mask = torch.tensor([True]*20,device=device)
    entries = torch.sum(mask)
    # print("huh:",mask.int().sub(2))
    print("mask:",mask)
    indices = torch.masked_fill(torch.cumsum(mask.int(),dim=0).sub(1), ~mask, a.shape[0])
    print("indices:",indices)
    result = torch.scatter(input=torch.zeros(a.shape[0]+1,dtype=a.dtype,device=device), dim=0, index=indices, src=a)[:entries]
    # torch.cuda.set_sync_debug_mode(0)

    print("result:",result)

def test_sync(device_id):
    device = torch.device(device_id)
    data = [torch.arange(0, (1<<24), 1, dtype=torch.int64,device=device) for _ in range(2)]
    streams = [torch.cuda.Stream(device=device) for _ in range(2)]
    sharder = ModuloSharding(4)
    # dummy call for initalization
    _ = sharder.shard_list_dynamic_mask([torch.arange(0, (1<<20), 1, dtype=torch.int64,device=device) for _ in range(1)], 0, device)
    _ = sharder.shard_in_place([torch.arange(0, (1<<20), 1, dtype=torch.int64,device=device) for _ in range(1)], 0, device)
    torch.cuda.synchronize(device=device)

    torch.cuda.nvtx.range_push(f"Test 1")
    with torch.cuda.stream(streams[0]):
        torch.cuda._sleep(100_000_000)
    with torch.cuda.stream(streams[1]):
        _ = sharder.shard_list_dynamic_mask(data,0,device)
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize(device=device)


    torch.cuda.nvtx.range_push(f"Test 2")
    with torch.cuda.stream(streams[0]):
        torch.cuda._sleep(100_000_000)
    with torch.cuda.stream(streams[1]):
        _ = sharder.shard_in_place(data,0,device)
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize(device=device)

    torch.cuda.nvtx.range_push(f"Test 3")
    with torch.cuda.stream(streams[1]):
        _ = sharder.shard_list_dynamic_mask(data,0,device)
    with torch.cuda.stream(streams[0]):
        torch.cuda._sleep(100_000_000)
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize(device=device)

    torch.cuda.nvtx.range_push(f"Test 4")
    with torch.cuda.stream(streams[1]):
        _ = sharder.shard_in_place(data,0,device)
    with torch.cuda.stream(streams[0]):
        torch.cuda._sleep(100_000_000)
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize(device=device)

def test_many_stream(device_id,num_streams):
    device = torch.device(device_id)
    data = [torch.arange(0, (1<<26), 1, dtype=torch.int64,device=device) for _ in range(2)]
    streams = [torch.cuda.Stream(device=device) for _ in range(num_streams)]
    sharder = ModuloSharding(4)

    torch.cuda.nvtx.range_push(f"MultiStream Test {num_streams=}")
    for round in range(16):
        for stream in streams:
            with torch.cuda.stream(stream):
                _ = sharder.shard_list_dynamic_mask(data,0,device)
                torch.cuda._sleep(100_000_000)
    torch.cuda.nvtx.range_pop()

    torch.cuda.synchronize(device=device)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <device_id>")
        exit(1)

    # always use rmm (not started with configured env in multiproc)
    # if os.environ['USE_RMM'] == "True":
    mr = rmm.mr.CudaAsyncMemoryResource()
    rmm.mr.set_current_device_resource(mr)
    torch.cuda.memory.change_current_allocator(rmm_torch_allocator)

    device_id = int(sys.argv[1])

    test(device_id, 2, 4, [256,512,1024])
    # test(device_id, 8, 1, [2])
    # test_unsync(device_id)
    # test_sync(device_id)
    # test_many_stream(device_id,2)
    # test_many_stream(device_id,4)
