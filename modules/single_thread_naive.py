import os
import time

import torch
import torch.nn as nn
import torch.distributed as dist

from helpers.util import hash_64, chunked_unique
from modules.common import SHUFFLE_PROBE_TYPE
from modules.partitioning_module import ModuloSharding
from modules.send_module import send_with_store,send_with_ccl

class ShuffleWithJoinNaive(nn.Module):
    """
    Current cecessary interface of left_child and right_child:
    get_dtypes() -> returns list of torch dtypes of columns
    get_morsel_count() -> returns Number of morsels
    () -> return list of column tensors, then None

    """
    def __init__(self, gpu, hash_table_init, left_child, left_index, right_child, right_index, store,
                    store_prefix="",distinct_ht=False,shuffle_left=True,project_build_side=None,keep_build_side=False,add_join_attrib=None):
        super().__init__()
        self._gpu = gpu
        self.world_rank = int(os.environ['RANK'])
        self.world_size = int(os.environ['WORLD_SIZE'])
        self.__sharder = ModuloSharding(self.world_size)

        self._hash_table = []
        # Hash table size configuration parameter
        self._hash_table_size = hash_table_init
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
        self.keep_build_side = keep_build_side
        self.add_join_attrib = add_join_attrib

        self.build_side_tuples = 0
        self.probe_side_tuples = 0
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
        self._matched_pairs = torch.zeros(1,dtype=torch.int64,device=self._gpu)
        self._probe_type = SHUFFLE_PROBE_TYPE[os.environ['SHUFFLE_PROBE_TYPE']]

    def name(self):
        return "ShuffleWithJoinNaive"

    def get_shuffled_bytes(self):
        left,right = 0,0
        build_tuples,probe_tuples,_ = self.get_stats()
        for dt in self._left_dtypes:
            left += build_tuples*dt.itemsize
        for dt in self._right_dtypes:
            right += probe_tuples*dt.itemsize
        return left,right

    def get_stats(self):
        return self.build_side_tuples,self.probe_side_tuples,self._matched_pairs.cpu().item()

    def forward(self, probe_pipeline=lambda *args, **kwargs: None):
        if probe_pipeline is None:
            # if the probe pipeline is None, the join tuples will still be collected, but nothing is done with them
            probe_pipeline = lambda *args, **kwargs: None

        stream = torch.cuda.Stream(device=self._gpu)
        torch.cuda.nvtx.range_push(f"Join. LocalMorsels: {[self.left_child.get_morsel_count(),self.right_child.get_morsel_count()]}")
        start = time.perf_counter_ns()

        # Left phase: Build hash table
        torch.cuda.nvtx.range_push("Left Side")

        morsel_counts = torch.tensor([self.left_child.get_morsel_count(),self.right_child.get_morsel_count()],
                                        dtype=torch.int64, device="cpu")
        # print("local morsel_counts:",morsel_counts)
        dist.all_reduce(morsel_counts, op=dist.ReduceOp.MAX)
        # print(f"Rounds (left/right): {morsel_counts}")

        # all data of the build side is materialized as usual
        build_side_cols_list = []
        build_side_cols = []
        left_hash_list = []
        left_hash = None
        step_time_ns = [0.0,0.0,0.0]

        # Run left shuffle
        for left_iter in range(morsel_counts[0]):
            with torch.cuda.stream(stream):
                before_part_event = torch.cuda.Event(enable_timing=True)
                before_part_event.record(stream)
                # Request new data for the next iteration (here to optimally parallelize copy in host scenario)
                data = self.left_child()
                if self.shuffle_left:
                    if data is not None:
                        if data[0].get_device() < 0:
                            target_data = [d.to(self._gpu) for d in data]
                        else:
                            target_data = data
                        partitioned_data = self.__sharder.shard_list_dynamic(target_data, self.left_index, self._gpu)
                    else:
                        partitioned_data = self._empty_left

                    before_shuffle_event = torch.cuda.Event(enable_timing=True)
                    before_shuffle_event.record(stream)
                    # Run shuffle
                    # nccl_data = self.__send_left(partitioned_data)
                    # nccl_data = send_with_store(self.__store, self.__store_prefix, self.world_rank, self.world_size,
                    #                             left_iter, partitioned_data, self._left_dtypes)
                    nccl_data = send_with_ccl(self.world_size, partitioned_data, self._left_dtypes)
                else: # -> self.shuffle_left = False
                    before_shuffle_event = torch.cuda.Event(enable_timing=True)
                    before_shuffle_event.record(stream)
                    if data is not None:
                        nccl_data = [data]
                    else:
                        nccl_data = [[torch.empty(0, dtype=dtype, device=self._gpu) for dtype in self._left_dtypes]]

                before_buildprobe_event = torch.cuda.Event(enable_timing=True)
                before_buildprobe_event.record(stream)
                # Extend the hash table with the newly received data
                for buffer in nccl_data:
                    # save the actual data
                    if self.project_build_side is not None:
                        build_side_cols_list.append([buffer[col_idx] for col_idx in self.project_build_side])
                    else:
                        build_side_cols_list.append(buffer)
                    if not self.distinct_ht:
                        # save the hashes of the left index column for the hashtable
                        left_hash_list.append(hash_64(buffer[self.left_index], self._hash_table_size))
                after_buildprobe_event = torch.cuda.Event(enable_timing=True)
                after_buildprobe_event.record(stream)

                stream.synchronize()
                step_time_ns[0] += (before_part_event.elapsed_time(before_shuffle_event)*1e6)
                step_time_ns[1] += (before_shuffle_event.elapsed_time(before_buildprobe_event)*1e6)
                step_time_ns[2] += (before_buildprobe_event.elapsed_time(after_buildprobe_event)*1e6)

        ### READER CHAINING LOGIC ###
        sequential_mode = os.environ.get("READER_SEQUENTIAL", "False").lower() == "true"
        if sequential_mode:
            data = self.left_child() # need to trigger finishing in FIFO
            assert data is None
        ##############################
        
        torch.cuda.nvtx.range_pop()
        left_shuffle_done = time.perf_counter_ns()
        torch.cuda.nvtx.range_push("Build HT")
        # Concatenate left remainders, compute left indices, compute maximum hash table size
        if self.distinct_ht:
            index_col = torch.cat(tuple(buffer[0] for buffer in build_side_cols_list))
            distinct_vals = torch.unique(index_col)
            # print(f"# Number of distinct vals: {distinct_vals.shape}")

            # construct full columns:
            build_side_cols.append(distinct_vals)
            left_hash = hash_64(distinct_vals, self._hash_table_size)
        else:
            left_hash = torch.cat(tuple(left_hash_list))
        bincount = left_hash.bincount()
        self._max_hash_bucket_size = torch.max(bincount).item() # this synchronizes
        print(f"# INFO: Number of hashtables: {self._max_hash_bucket_size}")
        del bincount
        if self.project_build_side is not None:
            # update left_index here to be zero (index columns in first). needed for probe
            self.left_index = 0

        if not self.distinct_ht:
            # construct all full columns:
            for col_idx in range(len(build_side_cols_list[0])):
                build_side_cols.append(torch.cat(tuple(buffer[col_idx] for buffer in build_side_cols_list)))
            self.build_side_tuples = build_side_cols[0].shape[0]

        # Compute hash table(s)
        for _ in range(self._max_hash_bucket_size):
            hash_table = torch.full((self._hash_table_size + 1,), -1, dtype=torch.int64, device=self._gpu)
            hash_table.scatter_(0, left_hash, torch.arange(0, left_hash.shape[0], dtype=torch.int64, device=self._gpu))
            left_index_select = hash_table[hash_table >= 0]
            left_hash[left_index_select] = self._hash_table_size
            self._hash_table.append(hash_table)

        torch.cuda.synchronize(device=self._gpu)
        build_side_cols_list = [] # empty
        left_hash_list = [] # empty
        left_hash = [] # empty
        torch.cuda.nvtx.range_pop()
        build_done = time.perf_counter_ns()
        step_time_ns[2] += (build_done - left_shuffle_done)
        torch.cuda.nvtx.range_push("Right Side")

        # Run right shuffle
        for right_iter in range(morsel_counts[1]):
            with torch.cuda.stream(stream):
                total_iter = right_iter + morsel_counts[0].item()
                # Check if we have local data, if so partition them
                before_part_event = torch.cuda.Event(enable_timing=True)
                before_part_event.record(stream)
                data = self.right_child()
                if data is not None:
                    if data[0].get_device() < 0:
                        target_data = [d.to(self._gpu) for d in data]
                    else:
                        target_data = data
                    partitioned_data = self.__sharder.shard_list_dynamic(target_data, self.right_index, self._gpu)
                else:
                    partitioned_data = self._empty_right

                before_shuffle_event = torch.cuda.Event(enable_timing=True)
                before_shuffle_event.record(stream)
                # Run shuffle
                # (store, store_prefix, world_rank, world_size, iteration, data, dtypes)
                # nccl_data = send_with_store(self.__store, self.__store_prefix, self.world_rank, self.world_size,
                #                                 total_iter, partitioned_data, self._right_dtypes)
                nccl_data = send_with_ccl(self.world_size, partitioned_data, self._right_dtypes)

                before_buildprobe_event = torch.cuda.Event(enable_timing=True)
                before_buildprobe_event.record(stream)

                for idx,buffer in enumerate(nccl_data):
                    self.__probe(build_side_cols, buffer, probe_pipeline)

                after_buildprobe_event = torch.cuda.Event(enable_timing=True)
                after_buildprobe_event.record(stream)

                stream.synchronize()
                step_time_ns[0] += (before_part_event.elapsed_time(before_shuffle_event)*1e6)
                step_time_ns[1] += (before_shuffle_event.elapsed_time(before_buildprobe_event)*1e6)
                step_time_ns[2] += (before_buildprobe_event.elapsed_time(after_buildprobe_event)*1e6)

        ### READER CHAINING LOGIC ###
        if sequential_mode:
            data = self.right_child() # need to trigger finishing in FIFO
            assert data is None
        ##############################
        
        torch.cuda.synchronize(device=self._gpu)
        torch.cuda.nvtx.range_pop()
        end = time.perf_counter_ns()
        torch.cuda.nvtx.range_pop()
        time_ns = end-start

        if self.keep_build_side:
            self.build_side_cols = build_side_cols

        return { "local_result": self._matched_pairs.cpu().item(), "time_ns": time_ns, "time_part_ns": step_time_ns[0],
                 "time_shuffle_ns": step_time_ns[1], "time_buildprobe_ns": step_time_ns[2],
                 "shuffled_bytes": self.get_shuffled_bytes(), "stats": self.get_stats()}


    def __probe(self, build_side_cols, probe_side_cols, probe_pipeline):
        if self._probe_type == SHUFFLE_PROBE_TYPE.NOOP:
            return
        left_index_col = build_side_cols[self.left_index]
        right_index_col = probe_side_cols[self.right_index]
        right_hash = hash_64(right_index_col, self._hash_table_size)
        self.probe_side_tuples += right_index_col.shape[0]

        for hash_table in self._hash_table:
            left_candidate_indices = hash_table[right_hash]
            match_mask = (left_candidate_indices>=0) & (left_index_col[left_candidate_indices] == right_index_col)
            if self.add_join_attrib is not None:
                for add_left_idx,add_right_idx in self.add_join_attrib:
                    match_mask &= build_side_cols[add_left_idx][left_candidate_indices] == probe_side_cols[add_right_idx]
            self._matched_pairs[0] += match_mask.sum()

            if self._probe_type == SHUFFLE_PROBE_TYPE.MATCHES_ONLY:
                pass
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

