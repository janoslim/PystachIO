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

class ShuffleWithJoin_MultiStream(nn.Module):
    def __init__(self, gpu, hash_table_init, left_child, left_index, right_child, right_index, store,
                    store_prefix="", distinct_ht=False,shuffle_left=True,project_build_side=None,keep_build_side=False,add_join_attrib=None):
        super().__init__()
        self._gpu = gpu
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
        self.left_child = left_child
        self.left_index = left_index
        self.right_child = right_child
        self.right_index = right_index
        self.distinct_ht = distinct_ht      # whether to create a ht of distinct keys of left side (e.g. for semijoins)
        self.shuffle_left = shuffle_left    # whether to shuffle the left side (e.g. False for asymmetrics re-partitioning)
        self.project_build_side = project_build_side    # projection build side columns after shuffling (e.g. for tpch query5)
        if self.project_build_side is not None:
            assert self.left_index == self.project_build_side[0]
        self.keep_build_side = keep_build_side    # keep build side columns after shuffling (e.g. for left outer join)
        self.add_join_attrib = add_join_attrib

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
        self._send_type = SHUFFLE_SEND_TYPE[os.environ['SHUFFLE_SEND_TYPE']]
        self._recv_type = SHUFFLE_RECV_TYPE[os.environ['SHUFFLE_RECV_TYPE']]
        self._probe_type = SHUFFLE_PROBE_TYPE[os.environ['SHUFFLE_PROBE_TYPE']]
        self._probe_sched = SHUFFLE_PROBE_SCHED[os.environ['SHUFFLE_PROBE_SCHED']]
        self._build_streaming = int(os.environ["SHUFFLE_BUILD_STREAMING_HT"])
        self._build_streaming_tup_offset = 0
        if self._build_streaming > 0:
            for _ in range(self._build_streaming):
                self._hash_table.append(torch.full((self._hash_table_size + 1,), -1, dtype=torch.int64, device=self._gpu))

        # differente cuda streams and groups
        self._shuffle_type = SHUFFLE_TYPE[os.environ['SHUFFLE_TYPE']]
        self._streams = [torch.cuda.Stream(device=self._gpu) for _ in range(int(os.environ['SHUFFLE_STREAMS']))]
        self._matched_pairs = torch.zeros(len(self._streams),dtype=torch.int64,device=self._gpu)
        self.build_side_tuples = 0
        self.probe_side_tuples = 0
        if self._shuffle_type == SHUFFLE_TYPE.OVERLAPPED_GROUPS:
            self._groups = [dist.new_group(timeout=timedelta(seconds=30), backend="cuda:nccl,cpu:gloo",
                                            group_desc=f"Group for Stream {streamidx}")
                                            for streamidx in range(int(os.environ['SHUFFLE_STREAMS']))]
            for group in self._groups:
                nccl_init_sync(self._gpu, group)
                # group.barrier()

            dist.barrier()
        else:
            self._groups = [None for _ in self._streams]

    def __del__(self):
        if self._shuffle_type == SHUFFLE_TYPE.OVERLAPPED_GROUPS:
            # clean up groups:
            for group in self._groups:
                dist.destroy_process_group(group)

    def name(self):
        if self._shuffle_type == SHUFFLE_TYPE.OVERLAPPED_GROUPS:
            return "ShuffleWithJoin_MultiGroup"
        else:
            return "ShuffleWithJoin_MultiStream"

    def get_shuffled_bytes(self):
        left,right = 0,0
        build_tuples,probe_tuples,_ = self.get_stats()
        for dt in self._left_dtypes:
            left += build_tuples*dt.itemsize
        for dt in self._right_dtypes:
            right += probe_tuples*dt.itemsize
        return left,right

    def get_stats(self):
        return self.build_side_tuples,self.probe_side_tuples,self._matched_pairs.sum().cpu().item()

    def forward(self, probe_pipeline=lambda *args, **kwargs: None):
        if probe_pipeline is None:
            # if the probe pipeline is None, the join tuples will still be collected, but nothing is done with them
            probe_pipeline = lambda *args, **kwargs: None

        htbuild_stream = torch.cuda.Stream(device=self._gpu)
        start_event = torch.cuda.Event(enable_timing=True)
        left_shuffle_done = torch.cuda.Event(enable_timing=True)
        build_done = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        torch.cuda.nvtx.range_push(f"Join. LocalMorsels: {[self.left_child.get_morsel_count(),self.right_child.get_morsel_count()]}")
        start = time.perf_counter_ns()
        # Left phase: Build hash table
        torch.cuda.nvtx.range_push("Left Side")
        start_event.record(self._streams[0])

        morsel_counts = torch.tensor([self.left_child.get_morsel_count(),self.right_child.get_morsel_count()],
                                        dtype=torch.int64, device="cpu")
        # print("local morsel_counts:",morsel_counts)
        dist.all_reduce(morsel_counts, op=dist.ReduceOp.MAX)
        print(f"# Rounds (left/right): {morsel_counts}")

        # all data of the build side is materialized as usual
        build_side_cols_list = []
        build_side_cols = []
        left_hash_list = []
        left_hash = None

        # events inserted after each morsel has been processed. used to synchronize with HT Build
        left_events = [None] * len(self._streams)
        ready_to_build = [None] * len(self._streams)

        # Run left shuffle
        for left_iter in range(morsel_counts[0]+len(self._streams)):
            streamidx = left_iter % len(self._streams)
            stream = self._streams[streamidx]
            # print(f"{left_iter=} done by stream {streamidx}")

            with torch.cuda.stream(stream):
                total_iter = left_iter

                if self._build_streaming > 0 and ready_to_build[streamidx] != None:
                    for buffer in ready_to_build[streamidx]:
                        left_num = buffer[self.left_index].numel()
                        done = 0
                        left_hash = hash_64(buffer[self.left_index],self._hash_table_size)
                        for hash_table in self._hash_table:
                            hash_table.scatter_(0, left_hash, torch.arange(self._build_streaming_tup_offset,
                                                                           self._build_streaming_tup_offset + left_num, dtype=torch.int64, device=self._gpu))
                            left_index_select = hash_table[hash_table >= self._build_streaming_tup_offset] # this synchronizes
                            left_index_select -= self._build_streaming_tup_offset
                            done += left_index_select.numel()
                            left_hash[left_index_select] = self._hash_table_size_device[0] # computation purely on device
                            # print(f"# Left side (size {left_num=}) streaming ht build {left_index_select.numel()}. {done=}, total_left={self._build_streaming_tup_offset}")
                            if done >= left_num:
                                break

                        self._build_streaming_tup_offset += left_num
                        # save the actual data
                        if self.project_build_side is not None:
                            build_side_cols_list.append([buffer[col_idx] for col_idx in self.project_build_side])
                        else:
                            build_side_cols_list.append(buffer)

                data = self.left_child() # FIFO dequeue
                if self.shuffle_left:
                    if data is not None:
                        if data[0].get_device() < 0:
                            target_data = [d.to(self._gpu) for d in data]
                        else:
                            target_data = data
                        if self._send_type == SHUFFLE_SEND_TYPE.SEND_PER_RANK:
                            partitioned_data = self.__sharder.shard_list_dynamic(target_data, self.left_index, self._gpu)
                        elif self._send_type == SHUFFLE_SEND_TYPE.COMB_SEND:
                            partitioned_data, input_splits = self.__sharder.shard_in_place(target_data, self.left_index, self._gpu)
                    else:
                        if self._send_type == SHUFFLE_SEND_TYPE.SEND_PER_RANK:
                            partitioned_data = self._empty_left
                        elif self._send_type == SHUFFLE_SEND_TYPE.COMB_SEND:
                            partitioned_data, input_splits = self._empty_left[0], torch.zeros(self.world_size, dtype=torch.int64, device="cpu")

                    # print(f"# Left transfer sizes (mb): {[(col.numel() * col.element_size())/(1<<20) for part_data in partitioned_data for col in part_data]}")

                    # shuffle
                    if self._send_type == SHUFFLE_SEND_TYPE.SEND_PER_RANK:
                        if self._recv_type == SHUFFLE_RECV_TYPE.RECV_PER_RANK:
                            nccl_data = send_with_ccl(self.world_size, partitioned_data, self._left_dtypes, group=self._groups[streamidx])
                        elif self._recv_type == SHUFFLE_RECV_TYPE.COMB_RECV:
                            nccl_data = [send_with_ccl_combine_recv(self.world_size, partitioned_data, self._left_dtypes, group=self._groups[streamidx])]
                    elif self._send_type == SHUFFLE_SEND_TYPE.COMB_SEND:
                        if self._recv_type == SHUFFLE_RECV_TYPE.RECV_PER_RANK:
                            nccl_data = send_inplace_with_ccl_per_rank(self.world_size, partitioned_data, input_splits, self._left_dtypes, group=self._groups[streamidx])
                        elif self._recv_type == SHUFFLE_RECV_TYPE.COMB_RECV:
                            nccl_data = [send_inplace_with_ccl(self.world_size, partitioned_data, input_splits, self._left_dtypes, group=self._groups[streamidx])]
                else: # -> self.shuffle_left = False
                    if data is not None:
                        nccl_data = [data]
                    else:
                        nccl_data = [[torch.empty(0, dtype=dtype, device=self._gpu) for dtype in self._left_dtypes]]

                # hash table prep
                if self._build_streaming == 0:
                    for buffer in nccl_data:
                        # save the actual data
                        if self.project_build_side is not None:
                            build_side_cols_list.append([buffer[col_idx] for col_idx in self.project_build_side])
                        else:
                            build_side_cols_list.append(buffer)
                        if not self.distinct_ht:
                            # save the hashes of the left index column for the hashtable
                            left_hash_list.append(hash_64(buffer[self.left_index],self._hash_table_size))
                else:
                    ready_to_build[streamidx] = nccl_data

                left_events[streamidx] = torch.cuda.Event()
                left_events[streamidx].record(stream)

        ### READER CHAINING LOGIC ###
        sequential_mode = os.environ.get("READER_SEQUENTIAL", "False").lower() == "true"
        if sequential_mode:
            data = self.left_child() # need to trigger finishing in FIFO
            assert data is None
        ##############################

        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("Right Side")
        ready_to_probe = [None] * len(self._streams)
        scheduled_htbuild = False


        # Run right shuffle
        for right_iter in range(morsel_counts[1]+len(self._streams)):
            total_iter = right_iter + morsel_counts[0].item() + len(self._streams)
            streamidx = total_iter % len(self._streams)
            stream = self._streams[streamidx]
            # print(f"{right_iter=}, {total_iter=} done by stream {streamidx}")

            with torch.cuda.stream(stream):
                if self._probe_sched == SHUFFLE_PROBE_SCHED.INVERTED and ready_to_probe[streamidx] != None:
                    # probe
                    build_done.wait(stream)
                    # torch.cuda.set_sync_debug_mode(1)
                    for idx,buffer in enumerate(ready_to_probe[streamidx]):
                        self.__probe(streamidx, build_side_cols, buffer, probe_pipeline)
                    # torch.cuda.set_sync_debug_mode(0)

                data = self.right_child()
                if data is not None:
                    if data[0].get_device() < 0:
                        target_data = [d.to(self._gpu) for d in data]
                    else:
                        target_data = data
                    if self._send_type == SHUFFLE_SEND_TYPE.SEND_PER_RANK:
                        partitioned_data = self.__sharder.shard_list_dynamic(target_data, self.right_index, self._gpu)
                    elif self._send_type == SHUFFLE_SEND_TYPE.COMB_SEND:
                        partitioned_data, input_splits = self.__sharder.shard_in_place(target_data, self.right_index, self._gpu)
                else:
                    if self._send_type == SHUFFLE_SEND_TYPE.SEND_PER_RANK:
                        partitioned_data = self._empty_right
                    elif self._send_type == SHUFFLE_SEND_TYPE.COMB_SEND:
                        partitioned_data, input_splits = self._empty_right[0], torch.zeros(self.world_size, dtype=torch.int64, device="cpu")

                # print(f"# Right transfer sizes (mb): {[(col.numel() * col.element_size())/(1<<20) for part_data in partitioned_data for col in part_data]}")

                # shuffle
                if self._send_type == SHUFFLE_SEND_TYPE.SEND_PER_RANK:
                    if self._recv_type == SHUFFLE_RECV_TYPE.RECV_PER_RANK:
                        nccl_data = send_with_ccl(self.world_size, partitioned_data, self._right_dtypes, group=self._groups[streamidx])
                    elif self._recv_type == SHUFFLE_RECV_TYPE.COMB_RECV:
                        nccl_data = [send_with_ccl_combine_recv(self.world_size, partitioned_data, self._right_dtypes, group=self._groups[streamidx])]
                elif self._send_type == SHUFFLE_SEND_TYPE.COMB_SEND:
                    if self._recv_type == SHUFFLE_RECV_TYPE.RECV_PER_RANK:
                        nccl_data = send_inplace_with_ccl_per_rank(self.world_size, partitioned_data, input_splits, self._right_dtypes, group=self._groups[streamidx])
                    elif self._recv_type == SHUFFLE_RECV_TYPE.COMB_RECV:
                        nccl_data = [send_inplace_with_ccl(self.world_size, partitioned_data, input_splits, self._right_dtypes, group=self._groups[streamidx])]

                ready_to_probe[streamidx] = nccl_data

                if not scheduled_htbuild:
                    if self._build_streaming > 0:
                        with torch.cuda.stream(htbuild_stream):
                            torch.cuda.nvtx.range_push("HT BUILD")
                            # first, wait for all possible left partitions
                            for left_event in left_events:
                                if left_event is None:
                                    continue
                                left_event.wait(torch.cuda.current_stream())
                            for col_idx in range(len(build_side_cols_list[0])):
                                build_side_cols.append(torch.cat(tuple(buffer[col_idx] for buffer in build_side_cols_list)))
                            build_done.record(torch.cuda.current_stream())
                            torch.cuda.nvtx.range_pop()
                    else: # self._build_streaming == 0
                        with torch.cuda.stream(htbuild_stream):
                            torch.cuda.nvtx.range_push("HT BUILD")
                            # first, wait for all possible left partitions
                            for left_event in left_events:
                                if left_event is None:
                                    continue
                                left_event.wait(torch.cuda.current_stream())
                            # then schedule the ht build stuff
                            left_shuffle_done.record(torch.cuda.current_stream())
                            # Concatenate left remainders, compute left indices, compute maximum hash table size

                            # DISTINCT Logic -> this is a pipeline breaker
                            if self.distinct_ht:
                                # distinct_vals = chunked_unique(build_side_cols_list, 0)
                                index_col = torch.cat(tuple(buffer[0] for buffer in build_side_cols_list))
                                distinct_vals = torch.unique(index_col)
                                # print(f"# Number of distinct vals: {distinct_vals.shape}")

                                # construct full columns:
                                build_side_cols.append(distinct_vals)
                                left_hash = hash_64(distinct_vals, self._hash_table_size)
                            else:
                                left_hash = torch.cat(tuple(left_hash_list))

                            bincount = torch.zeros((self._hash_table_size,), dtype=torch.int64,device=self._gpu)
                            bincount.scatter_reduce_(0, left_hash, torch.ones_like(left_hash), reduce="sum", include_self=False)
                            self._max_hash_bucket_size = torch.max(bincount).item() # this synchronizes, not pageable memory
                            del bincount
                            print(f"# INFO: Number of hashtables, size of hashtable, number of build side tuples: {self._max_hash_bucket_size}, {self._hash_table_size}, {left_hash.shape[0]}")
                            if self.project_build_side is not None:
                                # update left_index here to be zero (index columns in first). needed for probe
                                self.left_index = 0
                            left_hash_list.clear()

                            # Compute hash table(s)
                            for hash_table_idx in range(self._max_hash_bucket_size):
                                hash_table = torch.full((self._hash_table_size + 1,), -1, dtype=torch.int64, device=self._gpu)
                                hash_table.scatter_(0, left_hash, torch.arange(0, left_hash.shape[0], dtype=torch.int64, device=self._gpu))
                                self._hash_table.append(hash_table)
                                if hash_table_idx+1 < self._max_hash_bucket_size:
                                    left_index_select = hash_table[hash_table >= 0] # this synchronizes
                                    left_hash[left_index_select] = self._hash_table_size_device[0] # computation purely on device

                            if not self.distinct_ht:
                                # construct all full columns:
                                # for col_idx in range(len(self._left_dtypes)):
                                for col_idx in range(len(build_side_cols_list[0])):
                                    build_side_cols.append(torch.cat(tuple(buffer[col_idx] for buffer in build_side_cols_list)))
                                self.build_side_tuples = build_side_cols[0].shape[0]
                            # build_side_cols_list = [] # empty
                            # left_hash = [] # empty

                            build_done.record(torch.cuda.current_stream())
                            torch.cuda.nvtx.range_pop()
                    scheduled_htbuild = True

                if self._probe_sched == SHUFFLE_PROBE_SCHED.STANDARD:
                    # probe
                    build_done.wait(stream)
                    # torch.cuda.set_sync_debug_mode(1)
                    for idx,buffer in enumerate(ready_to_probe[streamidx]):
                        self.__probe(streamidx, build_side_cols, buffer, probe_pipeline)
                    # torch.cuda.set_sync_debug_mode(0)

        ### READER CHAINING LOGIC ###
        if sequential_mode:
            data = self.right_child() # need to trigger finishing in FIFO
            assert data is None
        ##############################
        
        end_event.record(self._streams[0])
        torch.cuda.synchronize(device=self._gpu)
        torch.cuda.nvtx.range_pop()
        end = time.perf_counter_ns()
        torch.cuda.nvtx.range_pop()
        time_ns = end-start

        if self.keep_build_side:
            self.build_side_cols = build_side_cols

        return { "local_result": self._matched_pairs.sum().cpu().item(), "time_ns": time_ns,
                 "shuffled_bytes": self.get_shuffled_bytes(), "stats": self.get_stats()}


    def __probe(self, streamidx, build_side_cols, probe_side_cols, probe_pipeline):
        if self._probe_type == SHUFFLE_PROBE_TYPE.NOOP:
            return
        torch.cuda.nvtx.range_push("HT PROBE")
        left_index_col = build_side_cols[self.left_index]
        right_index_col = probe_side_cols[self.right_index]
        right_hash = hash_64(right_index_col,self._hash_table_size)
        self.probe_side_tuples += right_index_col.shape[0]

        # joined = 0
        for hash_table in self._hash_table:
            left_candidate_indices = hash_table[right_hash]
            match_mask = (left_candidate_indices>=0) & (left_index_col[left_candidate_indices] == right_index_col)
            if self.add_join_attrib is not None:
                for add_left_idx,add_right_idx in self.add_join_attrib:
                    match_mask &= build_side_cols[add_left_idx][left_candidate_indices] == probe_side_cols[add_right_idx]
            self._matched_pairs[streamidx] += match_mask.sum()

            if self._probe_type == SHUFFLE_PROBE_TYPE.MATCHES_ONLY:
                # match_mask = (left_candidate_indices>=0) & (left_index_col[left_candidate_indices] == right_index_col)
                pass

            elif self._probe_type == SHUFFLE_PROBE_TYPE.FULL:
                as_indices = match_mask.nonzero().squeeze() # this synchronizes
                # joined += as_indices.numel()
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
                continue
        # if right_index_col.numel() != 0:
        #     print(f"Joined {joined} out of {right_index_col.numel()} right side tuples ({joined/right_index_col.numel():.4f})")
        torch.cuda.nvtx.range_pop()

