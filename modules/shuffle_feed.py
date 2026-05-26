import os
import time
import math
from datetime import timedelta

import torch
import torch.nn as nn
import torch.distributed as dist

from modules.common import SHUFFLE_TYPE,SHUFFLE_SEND_TYPE,SHUFFLE_RECV_TYPE,SHUFFLE_PROBE_TYPE,SHUFFLE_PROBE_SCHED
from helpers.util import hash_64, chunked_unique
from helpers.nccl_init import nccl_init_sync
from modules.partitioning_module import ModuloSharding
from modules.send_module import send_with_store,send_with_ccl,send_with_ccl_combine_recv,send_inplace_with_ccl,send_inplace_with_ccl_per_rank

class ShuffleFeed(nn.Module):
    def __init__(self, gpu, child, child_shuffle_idx):
        super().__init__()
        self._gpu = gpu
        self.world_rank = int(os.environ['RANK'])
        self.world_size = int(os.environ['WORLD_SIZE'])
        self.__sharder = ModuloSharding(self.world_size)

        # Metadata about the tables
        self.child = child
        self.child_shuffle_idx = child_shuffle_idx

        self.dtypes = child.get_dtypes()
        self.empty = [[torch.empty(0, dtype=dtype, device=self._gpu) for dtype in self.dtypes]
                                            for _ in range(self.world_size)]

        self._send_type = SHUFFLE_SEND_TYPE[os.environ['SHUFFLE_SEND_TYPE']]
        self._recv_type = SHUFFLE_RECV_TYPE[os.environ['SHUFFLE_RECV_TYPE']]
        self._probe_sched = SHUFFLE_PROBE_SCHED[os.environ['SHUFFLE_PROBE_SCHED']]

        # differente cuda streams and groups
        self._streams = [torch.cuda.Stream(device=self._gpu) for _ in range(int(os.environ['SHUFFLE_STREAMS']))]
        self.num_tuples = torch.zeros(len(self._streams),dtype=torch.int64,device=self._gpu)

        self._groups = [None for _ in self._streams]


    def name(self):
        return "ShuffleFeed"

    def get_shuffled_bytes(self):
        total = 0
        probe_tuples = self.get_stats()
        for dt in self.dtypes:
            total += probe_tuples*dt.itemsize
        return 0,total

    def get_stats(self):
        return self.num_tuples.sum().cpu().item()


    def forward(self, probe_pipeline=lambda *args, **kwargs: None):
        if probe_pipeline is None:
            probe_pipeline = lambda *args, **kwargs: None

        sequential_mode = os.environ.get("READER_SEQUENTIAL", "False").lower() == "true"
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        torch.cuda.nvtx.range_push(f"ShuffleFeed. LocalMorsels: {[self.child.get_morsel_count()]}")
        start = time.perf_counter_ns()
        start_event.record(self._streams[0])

        morsel_counts = torch.tensor([self.child.get_morsel_count()], dtype=torch.int64, device="cpu")
        # print("local morsel_counts:",morsel_counts)
        dist.all_reduce(morsel_counts, op=dist.ReduceOp.MAX)
        print(f"# ShuffleFeed Rounds : {morsel_counts}")

        # events inserted after each morsel has been processed. used to synchronize with HT Build
        ready_to_probe = [None] * len(self._streams)

        # Run shuffle feed
        for left_iter in range(morsel_counts[0]+len(self._streams)):
            streamidx = left_iter % len(self._streams)
            stream = self._streams[streamidx]
            # print(f"{left_iter=} done by stream {streamidx}")

            with torch.cuda.stream(stream):
                if self._probe_sched == SHUFFLE_PROBE_SCHED.INVERTED and ready_to_probe[streamidx] != None:
                    # probe
                    torch.cuda.nvtx.range_push("SF Probe")
                    for idx,buffer in enumerate(ready_to_probe[streamidx]):
                        self.num_tuples[streamidx] += buffer[0].shape[0]
                        probe_pipeline(buffer)
                    torch.cuda.nvtx.range_pop()

                data = self.child() # FIFO dequeue

                if data is not None:
                    if data[0].get_device() < 0:
                        target_data = [d.to(self._gpu) for d in data]
                    else:
                        target_data = data
                    if self._send_type == SHUFFLE_SEND_TYPE.SEND_PER_RANK:
                        partitioned_data = self.__sharder.shard_list_dynamic(target_data, self.child_shuffle_idx, self._gpu)
                    elif self._send_type == SHUFFLE_SEND_TYPE.COMB_SEND:
                        partitioned_data, input_splits = self.__sharder.shard_in_place(target_data, self.child_shuffle_idx, self._gpu)
                else:
                    if self._send_type == SHUFFLE_SEND_TYPE.SEND_PER_RANK:
                        partitioned_data = self.empty
                    elif self._send_type == SHUFFLE_SEND_TYPE.COMB_SEND:
                        partitioned_data, input_splits = self.empty[0], torch.zeros(self.world_size, dtype=torch.int64, device="cpu")

                # print(f"# Left transfer sizes (mb): {[(col.numel() * col.element_size())/(1<<20) for part_data in partitioned_data for col in part_data]}")

                # shuffle
                if self._send_type == SHUFFLE_SEND_TYPE.SEND_PER_RANK:
                    if self._recv_type == SHUFFLE_RECV_TYPE.RECV_PER_RANK:
                        nccl_data = send_with_ccl(self.world_size, partitioned_data, self.dtypes, group=self._groups[streamidx])
                    elif self._recv_type == SHUFFLE_RECV_TYPE.COMB_RECV:
                        nccl_data = [send_with_ccl_combine_recv(self.world_size, partitioned_data, self.dtypes, group=self._groups[streamidx])]
                elif self._send_type == SHUFFLE_SEND_TYPE.COMB_SEND:
                    if self._recv_type == SHUFFLE_RECV_TYPE.RECV_PER_RANK:
                        nccl_data = send_inplace_with_ccl_per_rank(self.world_size, partitioned_data, input_splits, self.dtypes, group=self._groups[streamidx])
                    elif self._recv_type == SHUFFLE_RECV_TYPE.COMB_RECV:
                        nccl_data = [send_inplace_with_ccl(self.world_size, partitioned_data, input_splits, self.dtypes, group=self._groups[streamidx])]

                ready_to_probe[streamidx] = nccl_data

                if self._probe_sched == SHUFFLE_PROBE_SCHED.STANDARD:
                    # probe
                    torch.cuda.nvtx.range_push("SF Probe")
                    for idx,buffer in enumerate(ready_to_probe[streamidx]):
                        self.num_tuples[streamidx] += buffer[0].shape[0]
                        probe_pipeline(buffer)
                    torch.cuda.nvtx.range_pop()

        ### READER CHAINING LOGIC ###
        if sequential_mode:
            data = self.child() # need to trigger finishing in FIFO
            assert data is None
        ##############################

        end_event.record(self._streams[0])
        torch.cuda.synchronize(device=self._gpu)
        end = time.perf_counter_ns()
        torch.cuda.nvtx.range_pop()
        time_ns = end-start

        return { "local_result": self.num_tuples.sum().cpu().item(), "time_ns": time_ns,
                 "shuffled_bytes": self.get_shuffled_bytes(), "stats": self.get_stats()}

