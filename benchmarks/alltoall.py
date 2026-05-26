import os
import sys
import time
from datetime import timedelta
import torch
import torch.distributed as dist

def test(target_device, world_size):
    size_bytes = 64 * (1<<20)
    rounds = 50
    send_tensors = [torch.arange(size_bytes, dtype=torch.int8, device=target_device) for _ in range(world_size)]
    recv_tensor = [torch.empty((size_bytes,), dtype=torch.int8, device=target_device) for _ in range(world_size)]

    stream = torch.cuda.Stream(device=target_device)

    torch.cuda.synchronize()
    dist.barrier()
    # init / warm up
    with torch.cuda.stream(stream):
        for _ in range(5):
            dist.all_to_all(input_tensor_list=send_tensors, output_tensor_list=recv_tensor)
    stream.synchronize()

    start = time.perf_counter_ns()
    with torch.cuda.stream(stream):
        for _ in range(rounds):
            dist.all_to_all(input_tensor_list=send_tensors, output_tensor_list=recv_tensor)
    stream.synchronize()
    end = time.perf_counter_ns()

    gb_per_sec = ((size_bytes*rounds*(world_size-1))/ (1<<30)) / ((end-start) / 1e9)
    print(f"Bus Bandwidth={gb_per_sec:.3f}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <target_device_id>")
        exit(1)
    target_device = torch.device(int(sys.argv[1]))
    WORLD_SIZE = int(os.environ['WORLD_SIZE'])
    WORLD_RANK = int(os.environ['RANK'])

    store = dist.TCPStore(os.environ['MASTER_ADDR'], int(os.environ['TCP_STORE_PORT'])+1, WORLD_SIZE,
                          WORLD_RANK == 0, timedelta(seconds=30))
    dist.init_process_group("cuda:nccl", store=store, rank=WORLD_RANK, world_size=WORLD_SIZE, device_id=target_device)

    test(target_device, WORLD_SIZE)

    dist.destroy_process_group()

