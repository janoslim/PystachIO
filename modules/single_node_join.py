import os
import time
import math

import torch
import torch.nn as nn

import sys
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
from helpers.util import hash_64

class SingleNodeJoin(nn.Module):
    def __init__(self, gpu, hash_table_init, left_child, right_child=None, right_col_idx=None, add_join_attrib=None,
                 anti_join=False):
        super().__init__()
        self._gpu = gpu

        # Hash table (list of tensors)
        self.hash_tables = []
        # build side data
        self.build_side_cols = []

        # Metadata about the tables
        self._left_dtypes = left_child.get_dtypes()
        # Hash table size configuration parameter
        self._hash_table_size = hash_table_init
        self._hash_table_size_device = torch.tensor([self._hash_table_size], dtype=torch.int64, device=self._gpu)
        self._magic_number = torch.tensor([5700357409661598721], dtype=torch.int64, device=self._gpu)
        # Maximum amount of entries with same hash (calculated dynamically)
        self._max_hash_bucket_size = 0
        self.num_join_tuples = 0
        self.build_side_tuples = 0
        self.probe_side_tuples = 0
        self.add_join_attrib = add_join_attrib
        self.anti_join = anti_join

        self._stream = torch.cuda.Stream(device=self._gpu)

        self.left_child = left_child
        if right_child is not None and right_col_idx is None:
            print("Error: SingleNodeJoin init: right_child is set, but right_col_idx is not")
        # only necessary for forward
        self.right_child = right_child
        self.right_col_idx = right_col_idx

    def get_stats(self):
        return self.build_side_tuples,self.probe_side_tuples,self.num_join_tuples

    def get_dtypes(self):
        # Dtypes of output tensors
        if self.right_child is None:
            print(f"Error: Called get_dtypes of SingleNodeJoin without setting the right child")
            return []
        return self.left_child.get_dtypes() + self.right_child.get_dtypes()


    def get_morsel_count(self):
        # Dtypes of output tensors
        if self.right_child is None:
            print(f"Error: Called get_morsel_count of SingleNodeJoin without setting the right child")
            return -1
        return self.right_child.get_morsel_count()


    def forward(self):
        """
        Convenience wrapper, only usable if right_child was given in constructor.
        """
        if self.right_child is None:
            print(f"Error: Called forward of SingleNodeJoin without setting the right child")
            return None

        torch.cuda.nvtx.range_push("SNJ Probe/forward")
        right_data = self.right_child()
        if right_data is None:
            torch.cuda.nvtx.range_pop()
            return None
        # concatenate for now:
        all_chunks = list(self.probe(right_data, self.right_col_idx))
        if len(all_chunks) == 0:
            # no join partners, just return empty
            torch.cuda.nvtx.range_pop()
            return [torch.empty(0, dtype=dtype, device=right_data[0].get_device()) for dtype in self.get_dtypes()]

        result = [torch.cat(tuple(chunk[col_idx] for chunk in all_chunks)) for col_idx in range(len(all_chunks[0]))]
        # Spill tree (patch 0003): right_data may be a reader-FIFO morsel allocated on a reader thread's stream. Once it
        # is dropped, cudaMallocAsync can hand its blocks to a reader's next decode while the probe gathers still read
        # them; the credit reader keeps readers decoding during the whole query, so let the gathers finish first.
        torch.cuda.current_stream().synchronize()
        torch.cuda.nvtx.range_pop()

        return result


    def build(self, left_child, left_index):
        """
        Only does the build side of the join, hence only reads left child
        TODO: for the broadcast, this should also use more than one stream
        """
        if id(left_child) != id(self.left_child):
            print(f"# Warn: SingleNodeJoin: Left child in init and forward are not the same!")
        self.left_index = left_index
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        torch.cuda.nvtx.range_push(f"SingleNodeJoin Morsels:{left_child.get_morsel_count()}")
        start = time.perf_counter_ns()
        start_event.record(self._stream)

        left_hash_list = []
        build_side_cols_list = []

        with torch.cuda.stream(self._stream):
            while (data := left_child()) is not None:
                build_side_cols_list.append(data)
                left_hash_list.append(hash_64(data[left_index], self._hash_table_size))

            # Concatenate left remainders, compute left indices, compute maximum hash table size
            left_hash = torch.cat(tuple(left_hash_list))
            bincount = torch.zeros((self._hash_table_size,), dtype=torch.int64, device=self._gpu)
            bincount.scatter_reduce_(0, left_hash, torch.ones_like(left_hash), reduce="sum", include_self=False)

            self._max_hash_bucket_size = torch.max(bincount).item() # this synchronizes, not pageable memory
            print(f"# INFO: SingleNodeJoin Number of hashtables, size of hashtable, number of build side tuples: {self._max_hash_bucket_size}, {self._hash_table_size}, {left_hash.shape[0]}")
            if self.anti_join and self._max_hash_bucket_size > 1:
                print(f"# WARN: Anti join with more than one hashtable not yet supported")
            # del bincount # free up a little memory?
            # Compute hash table(s)
            for hash_table_idx in range(self._max_hash_bucket_size):
                hash_table = torch.full((self._hash_table_size + 1,), -1, dtype=torch.int64, device=self._gpu)
                hash_table.scatter_(0, left_hash, torch.arange(0, left_hash.shape[0], dtype=torch.int64, device=self._gpu))
                self.hash_tables.append(hash_table)
                if hash_table_idx+1 < self._max_hash_bucket_size:
                    left_index_select = hash_table[hash_table >= 0] # this synchronizes
                    left_hash[left_index_select] = self._hash_table_size_device[0] # computation purely on device

            # construct all full columns:
            for col_idx in range(len(self._left_dtypes)):
                self.build_side_cols.append(torch.cat(tuple(buffer[col_idx] for buffer in build_side_cols_list)))
            self.build_side_tuples = self.build_side_cols[0].shape[0]
            # Spill tree (patch 0003): the build morsels may come from a reader FIFO, allocated on reader threads'
            # streams. Once they are dropped at return, cudaMallocAsync can hand their blocks to a reader's next
            # decode while the cat above still reads them, so let it finish first.
            self._stream.synchronize()

            # build_side_cols_list = [] # empty
            # left_hash_list = [] # empty
            # left_hash = [] # empty

        end_event.record(self._stream)
        torch.cuda.nvtx.range_pop()

        # return None

    def finish(self):
        """
            TODO: Fix misnomer, call finish_build
        """
        self._stream.synchronize()

    def release(self):
        """Spill tree (patch 0003): free the hash tables and build columns once nothing probes this join any more
        (GraceHashJoin.release_after_build_side). The last probes ran on the current stream: wait for them first."""
        torch.cuda.current_stream().synchronize()
        self.hash_tables = []
        self.build_side_cols = []

    def probe(self, probe_side_cols, right_index):
        left_index_col = self.build_side_cols[self.left_index]
        right_index_col = probe_side_cols[right_index]
        right_hash = hash_64(right_index_col, self._hash_table_size)
        self.probe_side_tuples += right_index_col.shape[0]

        for hash_table in self.hash_tables:
            left_candidate_indices = hash_table[right_hash]
            match_mask = (left_candidate_indices>=0) & (left_index_col[left_candidate_indices] == right_index_col)
            if self.add_join_attrib is not None:
                for add_left_idx,add_right_idx in self.add_join_attrib:
                    match_mask &= self.build_side_cols[add_left_idx][left_candidate_indices] == probe_side_cols[add_right_idx]
            if self.anti_join:
                torch.logical_not(match_mask,out=match_mask)
            as_indices = match_mask.nonzero().squeeze()
            if as_indices.numel() == 0:
                # early exit
                continue
            self.num_join_tuples += as_indices.numel()

            right_candidate_indices = torch.arange(0, right_index_col.shape[0], dtype=torch.int64, device=self._gpu)
            left_match_idx = torch.index_select(left_candidate_indices, 0, as_indices)
            right_match_idx = torch.index_select(right_candidate_indices, 0, as_indices)
            join_tuples = []
            for left_col_idx,left_column in enumerate(self.build_side_cols):
                join_tuples.append(left_column[left_match_idx])

            for right_col_idx,right_column in enumerate(probe_side_cols):
                join_tuples.append(right_column[right_match_idx])

            yield join_tuples


if __name__ == '__main__':
    from modules.morsel_table_module import MorselTableModule
    def simple_join():
        device = torch.device("cuda:0")
        build_cols = [torch.arange(0, 4500, dtype=torch.int64, device=device) for i in range(2)]
        probe_cols = [torch.arange(0, 9000, dtype=torch.int64, device=device) for i in range(2)]
        build_table = MorselTableModule(build_cols, 1000)
        probe_table = MorselTableModule(probe_cols, 500)
        join = SingleNodeJoin(device, 800, build_table, right_child=probe_table, right_col_idx=0)

        join.build(build_table, 0)
        join.finish()
        print(f"Number of hash_tables: {len(join.hash_tables)}")
        ######################################################
        # probe manually
        print(probe_cols)
        for data in join.probe(probe_cols, 0):
            # print(data)
            pass
        torch.cuda.synchronize(device=device)
        print(f"Number of probe results: {join.num_join_tuples=}")

        ######################################################
        # probe using forward and probe_table/right_child:
        join.num_join_tuples = 0
        print(f"{join.get_dtypes()=}, {join.get_morsel_count()=}")
        while (joined_tuples := join()) is not None:
            print(f"Joined once with probe chunk size: {joined_tuples[0].shape=}")
            torch.cuda.synchronize(device=device)
        print(f"Number of probe results: {join.num_join_tuples=}")

    def multi_attrib_join():
        device = torch.device("cuda:0")
        build_cols = [torch.tensor([0,1,0,1], dtype=torch.int64, device=device),
                      torch.tensor([2,5,4,3], dtype=torch.int64, device=device)]
        probe_cols = [torch.tensor([0,0,1,1], dtype=torch.int64, device=device),
                      torch.tensor([0,2,3,1], dtype=torch.int64, device=device),
                      torch.tensor([0,1,1,0], dtype=torch.int64, device=device),]
        # joined should be:
        # (0,2,0,2,1), (1,3,1,3,1)
        build_table = MorselTableModule(build_cols, 1000)
        probe_table = MorselTableModule(probe_cols, 2)
        join = SingleNodeJoin(device, 800, build_table, right_child=probe_table, right_col_idx=0,
                                add_join_attrib=[(1,1)])

        join.build(build_table, 0)
        join.finish()
        print(f"Number of hash_tables: {len(join.hash_tables)}")
        ######################################################
        # probe manually
        # print(probe_cols)
        for data in join.probe(probe_cols, 0):
            print(data)
        torch.cuda.synchronize(device=device)
        print(f"Join stats: {join.get_stats()}")

    # simple_join()
    multi_attrib_join()
