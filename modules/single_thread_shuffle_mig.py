import os
import time
import math
from datetime import datetime, timedelta
import torch.multiprocessing as mp
import sys
from queue import Empty

import torch
import torch.nn as nn
import torch.distributed as dist

from modules.common import SHUFFLE_TYPE,SHUFFLE_RECV_TYPE,SHUFFLE_PROBE_TYPE,SHUFFLE_PROBE_SCHED
from helpers.util import hash_64
from helpers.nccl_init import nccl_init_sync
from modules.partitioning_module import ModuloSharding
from modules.send_module import send_with_store,send_with_ccl,send_with_ccl_combine_recv


def _send_proc(world_rank, world_size, num_streams, p2c, c2p, left_dtypes, right_dtypes):
    sys.stdout = open(f"child_out{world_rank}.out", "a")
    sys.stderr = open(f"child_out{world_rank}.out", "a")
    print(f"######### Started {str(datetime.now())}")

    c2p.cancel_join_thread()
    target_device = torch.device("cuda:0")
    print("Cur_device in child:",torch.cuda.get_device_name(), os.environ["CUDA_VISIBLE_DEVICES"])
    store = dist.TCPStore(os.environ['MASTER_ADDR'], int(os.environ['TCP_STORE_PORT'])+1, world_size, world_rank == 0, timedelta(seconds=30))
    dist.init_process_group("cuda:nccl,cpu:gloo", store=store, rank=world_rank, world_size=world_size, device_id=target_device)
    nccl_init_sync(0)
    torch.cuda.synchronize()
    streams = [torch.cuda.Stream(device=target_device) for _ in range(num_streams)]
    queue = []

    c2p.put(0)  # notify parent child is ready

    total_left_iter,total_right_iter = p2c.get()

    total_iter = 0
    sent_next = 0
    while True:
        obj = None
        try:
            obj = p2c.get(block=False)
        except Empty as e:
            if total_iter >= num_streams and sent_next < len(queue):
                streams[sent_next%num_streams].synchronize()
                c2p.put((sent_next%num_streams,sent_next<total_left_iter,queue[sent_next]))
                sent_next += 1
            continue
        if obj == "DONE":
            break

        total_iter += 1
        print(f"{total_iter=}")
        streamidx, data, side = obj

        if total_iter >= num_streams and sent_next < len(queue):
            streams[sent_next%num_streams].synchronize()
            c2p.put((sent_next%num_streams,sent_next<total_left_iter,queue[sent_next]))
            sent_next += 1

        with torch.cuda.stream(streams[streamidx]):
            queue.append(send_with_ccl(world_size, data, left_dtypes if side else right_dtypes, target_device=target_device))

    print(f"Done in _send_proc, waiting for done in main proc ...")
    p2c.get()
    print(f"Destroying dis process group")
    dist.destroy_process_group()

class ShuffleWithJoin_MIG(nn.Module):
    def __init__(self, cpu, gpu, hash_table_init, left_child, right_child, store, store_prefix=""):
        super().__init__()
        self._cpu = cpu
        self._gpu = torch.device("cuda:0")
        self.world_rank = int(os.environ['RANK'])
        self.world_size = int(os.environ['WORLD_SIZE'])
        self.__sharder = ModuloSharding(self.world_size)

        # Hash table (list of tensors)
        self._hash_table = []
        # Hash table size configuration parameter
        self._hash_table_size = hash_table_init
        self._hash_table_size_device = torch.tensor([self._hash_table_size], dtype=torch.int64, device=self._gpu)
        self._magic_number = torch.tensor([5700357409661598721], dtype=torch.int64, device=self._gpu)
        # Maximum amount of entries with same hash (calculated dynamically)
        self._max_hash_bucket_size = 0
        # Metadata about the tables
        self._left_dtypes = left_child.get_dtypes()
        self._right_dtypes = right_child.get_dtypes()
        self._empty_left = [[torch.empty(0, dtype=dtype, device=self._gpu) for dtype in self._left_dtypes]
                                            for _ in range(self.world_size)]
        self._empty_right = [[torch.empty(0, dtype=dtype, device=self._gpu) for dtype in self._right_dtypes]
                                            for _ in range(self.world_size)]
        # Distributed Key-Value Store for exchanging metadata (length information)
        # store_prefix is prepended before every operation (e.g. use for multiple experiments with same store)
        self.__store = store
        self.__store_prefix = store_prefix
        # Count how many matching pairs we have found
        self._recv_type = SHUFFLE_RECV_TYPE[os.environ['SHUFFLE_RECV_TYPE']]
        self._probe_type = SHUFFLE_PROBE_TYPE[os.environ['SHUFFLE_PROBE_TYPE']]
        self._probe_sched = SHUFFLE_PROBE_SCHED[os.environ['SHUFFLE_PROBE_SCHED']]

        # different cuda streams and groups
        self._shuffle_type = SHUFFLE_TYPE[os.environ['SHUFFLE_TYPE']]
        self._streams = [torch.cuda.Stream(device=self._gpu) for _ in range(int(os.environ['SHUFFLE_STREAMS']))]
        self._matched_pairs = torch.zeros(len(self._streams),dtype=torch.int64,device=self._gpu)

        self.ctx = mp.get_context("spawn")
        self.p2c = self.ctx.Queue()
        self.c2p = self.ctx.Queue()
        tmp = os.environ["CUDA_VISIBLE_DEVICES"]
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"].split(",")[1]
        self.p = self.ctx.Process(
            target=_send_proc,
            args=(self.world_rank, self.world_size, len(self._streams), self.p2c, self.c2p, self._left_dtypes, self._right_dtypes),
        )
        self.p.start()
        os.environ["CUDA_VISIBLE_DEVICES"] = tmp # reset
        self.c2p.get()  # wait until child process is ready

    def name(self):
        return "ShuffleWithJoin_MIG"


    def forward(self, left_child, right_child, left_index, right_index, probe_pipeline=lambda *args, **kwargs: None):
        if probe_pipeline is None:
            # if the probe pipeline is None, the join tuples will still be collected, but nothing is done with them
            probe_pipeline = lambda *args, **kwargs: None

        htbuild_stream = torch.cuda.Stream(device=self._gpu)
        start_event = torch.cuda.Event(enable_timing=True)
        left_shuffle_done = torch.cuda.Event(enable_timing=True)
        build_done = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        open_left,open_right = 0,0

        torch.cuda.nvtx.range_push(f"Join. LocalMorsels: {[left_child.get_morsel_count(),right_child.get_morsel_count()]}")
        start = time.perf_counter_ns()
        # Left phase: Build hash table
        torch.cuda.nvtx.range_push("Left Side")
        start_event.record(self._streams[0])

        morsel_counts = torch.tensor([left_child.get_morsel_count(),right_child.get_morsel_count()],
                                        dtype=torch.int64, device="cpu")
        print("local morsel_counts:",morsel_counts)
        dist.all_reduce(morsel_counts, op=dist.ReduceOp.MAX)
        # print(f"Rounds (left/right): {morsel_counts}")

        # all data of the build side is materialized as usual
        build_side_cols_list = []
        build_side_cols = []
        left_hash_list = []
        left_hash = None

        # events inserted after each morsel has been processed. used to synchronize with HT Build
        left_events = [None] * len(self._streams)
        self.p2c.put((morsel_counts[0].item(),morsel_counts[1].item())) # communicate morsel counts to sender process

        c2p_gets = 0

        # Run left shuffle
        for left_iter in range(morsel_counts[0]):
            streamidx = left_iter % len(self._streams)
            stream = self._streams[streamidx]
            # print(f"{left_iter=} done by stream {streamidx}")

            with torch.cuda.stream(stream):
                total_iter = left_iter

                data = left_child() # FIFO dequeue
                if data is not None:
                    if data[0].get_device() < 0:
                        target_data = [d.to(self._gpu) for d in data]
                    else:
                        target_data = data
                    partitioned_data = self.__sharder.shard_list_dynamic(target_data, left_index, self._gpu)
                else:
                    partitioned_data = self._empty_left

                stream.synchronize()
                self.p2c.put((streamidx,partitioned_data,True))

                try:
                    cur_streamidx, side, nccl_data = self.c2p.get(block=False)
                    c2p_gets += 1
                    if not side:
                        print("Should not happen")
                    # pre hash
                    with torch.cuda.stream(self._streams[cur_streamidx]):
                        for buffer in nccl_data:
                            # save the actual data
                            build_side_cols_list.append(buffer)
                            # save the hashes of the left index column for the hashtable
                            left_hash_list.append(hash_64(buffer[left_index],self._hash_table_size))

                        left_events[streamidx] = torch.cuda.Event()
                        left_events[streamidx].record(stream)
                except Empty  as e:
                    pass

        torch.cuda.nvtx.range_pop()


        torch.cuda.nvtx.range_push("Right Side")
        scheduled_htbuild = False

        # Run right shuffle
        for right_iter in range(morsel_counts[1]):
            total_iter = right_iter + morsel_counts[1]+morsel_counts[1]%len(self._streams)
            streamidx = total_iter % len(self._streams)
            stream = self._streams[streamidx]
            # print(f"{right_iter=}, {total_iter=} done by stream {streamidx}")

            with torch.cuda.stream(stream):
                try:
                    cur_streamidx, side, nccl_data = self.c2p.get(block=False)
                    c2p_gets += 1
                    with torch.cuda.stream(self._streams[cur_streamidx]):
                        if side:
                            # pre hash
                            for buffer in nccl_data:
                                # save the actual data
                                build_side_cols_list.append(buffer)
                                # save the hashes of the left index column for the hashtable
                                left_hash_list.append(hash_64(buffer[left_index],self._hash_table_size))

                            left_events[streamidx] = torch.cuda.Event()
                            left_events[streamidx].record(stream)
                        else:
                            # build_done.wait(stream)
                            # for idx,buffer in enumerate(nccl_data):
                            #     self.__probe(streamidx, build_side_cols, left_index, buffer, right_index, probe_pipeline)
                            pass
                except Empty as e:
                    pass

                data = right_child()
                if data is not None:
                    if data[0].get_device() < 0:
                        target_data = [d.to(self._gpu) for d in data]
                    else:
                        target_data = data
                    partitioned_data = self.__sharder.shard_list_dynamic(target_data, right_index, self._gpu)
                else:
                    partitioned_data = self._empty_right

                stream.synchronize()
                self.p2c.put((streamidx,partitioned_data,False))

                # if not scheduled_htbuild:
                #     with torch.cuda.stream(htbuild_stream):
                #         torch.cuda.nvtx.range_push("HT BUILD")
                #         # first, wait for all possible left partitions
                #         for left_event in left_events:
                #             if left_event is None:
                #                 continue
                #             left_event.wait(torch.cuda.current_stream())
                #         # then schedule the ht build stuff
                #         left_shuffle_done.record(torch.cuda.current_stream())
                #         # Concatenate left remainders, compute left indices, compute maximum hash table size

                #         left_hash = torch.cat(tuple(left_hash_list))
                #         bincount = torch.zeros((self._hash_table_size,), dtype=torch.int64,device=self._gpu)
                #         bincount.scatter_reduce_(0, left_hash, torch.ones_like(left_hash), reduce="sum", include_self=False)

                #         self._max_hash_bucket_size = torch.max(bincount).item() # this synchronizes, not pageable memory
                #         # Compute hash table(s)
                #         for hash_table_idx in range(self._max_hash_bucket_size):
                #             hash_table = torch.full((self._hash_table_size + 1,), -1, dtype=torch.int64, device=self._gpu)
                #             hash_table.scatter_(0, left_hash, torch.arange(0, left_hash.shape[0], dtype=torch.int64, device=self._gpu))
                #             self._hash_table.append(hash_table)
                #             if hash_table_idx+1 < self._max_hash_bucket_size:
                #                 left_index_select = hash_table[hash_table >= 0] # this synchronizes
                #                 left_hash[left_index_select] = self._hash_table_size_device[0] # computation purely on device

                #         # construct all full columns:
                #         for col_idx in range(len(self._left_dtypes)):
                #             build_side_cols.append(torch.cat(tuple(buffer[col_idx] for buffer in build_side_cols_list)))

                #         # build_side_cols_list = [] # empty
                #         # left_hash_list = [] # empty
                #         # left_hash = [] # empty

                #         build_done.record(torch.cuda.current_stream())
                #         torch.cuda.nvtx.range_pop()
                #     scheduled_htbuild = True

        while c2p_gets != morsel_counts.sum().item():
            cur_streamidx, side, nccl_data = self.c2p.get()
            c2p_gets += 1
            with torch.cuda.stream(self._streams[cur_streamidx]):
                if side:
                    # pre hash
                    for buffer in nccl_data:
                        # save the actual data
                        build_side_cols_list.append(buffer)
                        # save the hashes of the left index column for the hashtable
                        left_hash_list.append(hash_64(buffer[left_index],self._hash_table_size))

                    left_events[streamidx] = torch.cuda.Event()
                    left_events[streamidx].record(stream)
                else:
                    # build_done.wait(stream)
                    # for idx,buffer in enumerate(nccl_data):
                    #     self.__probe(streamidx, build_side_cols, left_index, buffer, right_index, probe_pipeline)
                    pass
        print("SENTINEL will be sent in producer", flush=True)
        self.p2c.put("DONE")
        self.p2c.put(0)
        # todo: need to wait here for a sync from client? not really probably
        # self.c2p.get()
        # print("SENTINEL was synced back in producer", flush=True)

        end_event.record(self._streams[0])
        torch.cuda.synchronize(device=self._gpu)
        torch.cuda.nvtx.range_pop()
        end = time.perf_counter_ns()
        torch.cuda.nvtx.range_pop()
        time_ns = end-start

        self.p.join()
        print("Joined p", flush=True)

        # left_bus_bw_gibs = ((left_child.bytes())*((self.world_size-1)/self.world_size) / (1<<30)) / ((start_event.elapsed_time(left_shuffle_done)) / 1e3)
        # right_bus_bw_gibs = ((right_child.bytes())*((self.world_size-1)/self.world_size) / (1<<30)) / ((build_done.elapsed_time(end_event)) / 1e3)

        # print(f"Left side shuffle   : {start_event.elapsed_time(left_shuffle_done):.6f}ms")
        # print(f"Left side bus bw    : {left_bus_bw_gibs:.6f}GiB/s")
        # print(f"Build side          : {left_shuffle_done.elapsed_time(build_done):.6f}ms")
        # print(f"Right side          : {build_done.elapsed_time(end_event):.6f}ms")
        # print(f"Right side bus bw   : {right_bus_bw_gibs:.6f}GiB/s")
        # print(f"Total time          : {time_ns/1e6:.6f}ms")

        return { "local_result": self._matched_pairs.sum().cpu().item(), "time_ns": time_ns}


    def __probe(self, streamidx, build_side_cols, left_index, probe_side_cols, right_index, probe_pipeline):
        torch.cuda.nvtx.range_push("HT PROBE")
        left_index_col = build_side_cols[left_index]
        right_index_col = probe_side_cols[right_index]
        right_hash = hash_64(right_index_col,self._hash_table_size)

        for hash_table in self._hash_table:
            left_candidate_indices = hash_table[right_hash]
            match_mask = (left_candidate_indices>=0) & (left_index_col[left_candidate_indices] == right_index_col)
            self._matched_pairs[streamidx] += match_mask.sum()

            if self._probe_type == SHUFFLE_PROBE_TYPE.MATCHES_ONLY:
                # match_mask = (left_candidate_indices>=0) & (left_index_col[left_candidate_indices] == right_index_col)
                pass
            elif self._probe_type == SHUFFLE_PROBE_TYPE.MASKED:
                # match_mask = (left_candidate_indices>=0) & (left_index_col[left_candidate_indices] == right_index_col)
                # print(f"WARN: MASKED probe type not implemented yet")
                # todo: new probe pipeline signature could look like:
                # probe_pipeline(build_side_cols,left_candidate_indices,probe_side_cols,match_mask)
                # or rather
                data = [col[left_candidate_indices] for col in build_side_cols] + probe_side_cols
                probe_pipeline(data,match_mask)
            elif self._probe_type == SHUFFLE_PROBE_TYPE.FULL:
                as_indices = match_mask.nonzero().squeeze() # this synchronizes
                if as_indices.numel() == 0:
                    # early exit
                    continue

                right_indices = torch.arange(0, right_index_col.shape[0], dtype=torch.int64, device=self._gpu)
                left_match_idx = torch.index_select(left_candidate_indices, 0, as_indices)
                right_match_idx = torch.index_select(right_indices, 0, as_indices)
                data = []
                for left_col_idx,left_column in enumerate(build_side_cols):
                    data.append(left_column[left_match_idx])

                for right_col_idx,right_column in enumerate(probe_side_cols):
                    data.append(right_column[right_match_idx])
                # anything else is configurable through the probe_pipeline.
                # every probe pipeline should accept the data, indices and the match mask
                probe_pipeline(data)
            else:
                print(f"ERROR: Unknown probe type: {self._probe_type}")
                return
        torch.cuda.nvtx.range_pop()

