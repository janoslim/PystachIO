import os
import sys
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
import time

import torch
import torch.nn as nn
import torch.distributed as dist

from modules.common import SHUFFLE_RECV_TYPE
from modules.send_module import send_with_store,send_with_ccl,broadcast_ccl

class Broadcast(nn.Module):
    """
    Current cecessary interface of left_child and right_child:
    get_dtypes() -> returns list of torch dtypes of columns
    get_morsel_count() -> returns Number of morsels
    () -> return list of column tensors, then None

    """
    def __init__(self, gpu, child):
        super().__init__()
        self._gpu = gpu
        self.world_rank = int(os.environ['RANK'])
        self.world_size = int(os.environ['WORLD_SIZE'])

        self._recv_type = SHUFFLE_RECV_TYPE[os.environ['SHUFFLE_RECV_TYPE']]

        # Metadata about the tables
        self.child = child
        self.dtypes = child.get_dtypes()
        self.empty = [torch.empty(0, dtype=dtype, device=self._gpu) for dtype in self.dtypes]


    def forward(self):
        if torch.cuda.current_stream(self._gpu) == torch.cuda.default_stream(self._gpu):
            print(f"WARN: Rank[{self.world_rank}]: Broadcast forward called in default stream, probably not wanted")

        # morsel_counts = torch.tensor([self.child.get_morsel_count()], dtype=torch.int64, device="cpu")
        # print("local morsel_counts:",morsel_counts)
        # dist.all_reduce(morsel_counts, op=dist.ReduceOp.MAX)
        # for morsel_iter in range(morsel_counts[0]):

        data = self.child()
        if data is None:
            data = self.empty

        if self._recv_type == SHUFFLE_RECV_TYPE.RECV_PER_RANK:
            recv_data = broadcast_ccl(self.world_size, data, self.dtypes, self._gpu)
        elif self._recv_type == SHUFFLE_RECV_TYPE.COMB_RECV:
            recv_data = [broadcast_ccl_combine_recv(self.world_size, data, self.dtypes, self._gpu)]

        # exit only if no rank had any data to share:
        if all(cols[0].numel() == 0 for cols in recv_data):
            # print(f"End of broadcast")
            return None

        if self._recv_type == SHUFFLE_RECV_TYPE.RECV_PER_RANK:
            # recv_data = [[col0Tensor, ...], [col0Tensor, ...], ...]
            cat_data = [torch.cat(tuple(cols[col_idx] for cols in recv_data)) for col_idx,_ in enumerate(self.dtypes)]
            # cat_data = [col0Tensor, ... ]
            return cat_data
        elif self._recv_type == SHUFFLE_RECV_TYPE.COMB_RECV:
            return recv_data[0]


if __name__ == '__main__':
    from modules.morsel_table_module import MorselTableModule
    from datetime import timedelta

    WORLD_SIZE = int(os.environ['WORLD_SIZE'])
    WORLD_RANK = int(os.environ['RANK'])
    MASTER_IP = os.environ['MASTER_ADDR']
    TCP_STORE_PORT = int(os.environ['TCP_STORE_PORT'])
    target_device = int(sys.argv[1])

    # store = DummyStore(MASTER_IP)
    store = dist.TCPStore(MASTER_IP, TCP_STORE_PORT, WORLD_SIZE, WORLD_RANK == 0, timedelta(seconds=30))
    dist.init_process_group("cuda:nccl,cpu:gloo", store=store, rank=WORLD_RANK, world_size=WORLD_SIZE, device_id=torch.device(target_device))

    os.environ['SHUFFLE_RECV_TYPE'] = "RECV_PER_RANK" 
    # os.environ['SHUFFLE_RECV_TYPE'] = "COMB_RECV" 

    device = torch.device("cuda:0")
    cols = [torch.arange((WORLD_RANK)*5, (WORLD_RANK+1)*5, dtype=torch.int64, device=device) for i in range(2)]
    table = MorselTableModule(cols, 5)
    bc = Broadcast(device, table)

    ######################################################
    while (data := bc()) is not None:
        print(f"{data[0]}")
    torch.cuda.synchronize(device=device)

    dist.destroy_process_group()
