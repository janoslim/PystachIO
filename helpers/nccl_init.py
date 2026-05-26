import os
import torch
import torch.distributed as dist

from modules.partitioning_module import ModuloSharding

LOCAL_RANK = int(os.environ['LOCAL_RANK'])
WORLD_SIZE = int(os.environ['WORLD_SIZE'])
WORLD_RANK = int(os.environ['RANK'])


def nccl_init(device):
    buffers = [torch.tensor((1,), dtype=torch.int8, device=device) for _ in range(WORLD_SIZE)]
    mod_buffers = [i.remainder(WORLD_SIZE) for i in buffers]
    sharder = ModuloSharding(WORLD_SIZE)
    sharder.shard_list_dynamic(buffers, 0, device)
    ops = []
    for i in range(WORLD_SIZE):
        if not i == WORLD_RANK:
            ops.append(dist.P2POp(dist.isend, mod_buffers[i], i))
            ops.append(dist.P2POp(dist.irecv, buffers[i], i))
    data_requests = dist.batch_isend_irecv(ops)
    for req in data_requests:
        req.wait()

# def nccl_init_sync(device):
#     buffers = [torch.arange(0, 1000, dtype=torch.int8, device=device) for _ in range(WORLD_SIZE)]
#     mod_buffers = [i.remainder(WORLD_SIZE) for i in buffers]
#     sharder = ModuloSharding(WORLD_SIZE)
#     sharder.shard_list_dynamic(buffers, 0, device)
#     for i in range(WORLD_SIZE):
#         if not i == WORLD_RANK:
#             if i < WORLD_RANK:
#                 dist.send(mod_buffers[i], i)
#             dist.recv(buffers[i], i)
#             if i > WORLD_RANK:
#                 dist.send(mod_buffers[i], i)
#     torch.cuda.synchronize(device)

# Moved to p2p to get rid of bugged process group shutdown
# https://discuss.pytorch.org/t/multi-node-error-on-process-destruction-cuda-error-invalid-device-ordinal/211990
def nccl_init_sync(device, group=None):
    buffers = [torch.arange(0, 1000, dtype=torch.int8, device=device) for _ in range(WORLD_SIZE)]
    mod_buffers = [i.remainder(WORLD_SIZE) for i in buffers]
    sharder = ModuloSharding(WORLD_SIZE)
    sharder.shard_list_dynamic(buffers, 0, device)

    ops = []
    for i in range(WORLD_SIZE):
        if not i == WORLD_RANK:
            if i < WORLD_RANK:
                # dist.send(mod_buffers[i], i)
                ops.append(dist.P2POp(dist.isend, mod_buffers[i], i, group=group))
            # dist.recv(buffers[i], i)
            ops.append(dist.P2POp(dist.irecv, buffers[i], i, group=group))
            if i > WORLD_RANK:
                # dist.send(mod_buffers[i], i)
                ops.append(dist.P2POp(dist.isend, mod_buffers[i], i, group=group))
    data_requests = dist.batch_isend_irecv(ops)
    for req in data_requests:
        req.wait()
    torch.cuda.synchronize(device)
