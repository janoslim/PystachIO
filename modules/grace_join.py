"""Hybrid Grace hash join with SSD spill partitions: the join of the spill mode (SHUFFLE_TYPE=GRACE, patch 0003).

Drop-in for ShuffleWithJoin_MultiStream in Q3, Q5, Q10, Q12, Q14 and Q19: same constructor arguments, forward()
contract and statistics, plus the spill directories and the joined columns the query's probe pipeline reads. The
shuffle join builds direct-mapped hash tables over the whole build side, sized by the scale factor (H slots, e.g.
1.5e9 for Q3 at SF1000), which do not fit in GPU memory at SF1000. This join splits both sides by join key % P
(helpers/spill_io.py); partitions 0..k-1 stay in GPU memory, the other P-k go to the data SSDs:
1. drains the build side (left child): partitions below SPILL_RESIDENT_PARTITIONS are kept in GPU memory, the
   others written to spill files (their keys kept for the filter); then frees the modules in
   release_after_build_side (Q3/Q5/Q10's customer join);
2. chooses k at run time, when every kept partition's rows and hash tables are known: builds a SingleNodeJoin
   (next_prime(H / P) slots) on partitions 0, 1, ... while the next one's build peak, the filter over the partitions
   after it and the probe phase's reserve fit in the GPU memory still available, and writes the kept rows of the
   partitions from k on; then builds a filter over the build keys of the spilled partitions k..P-1
   (helpers/key_filter.py: exact bitmap or Bloom filter; none when k = P);
3. drains the probe side (right child): one kernel per batch computes the partition ids and drops the rows of
   spilled partitions whose key the filter rejects (they cannot join); resident partitions' rows are probed at once,
   the kept rows of the others are held and written a batch's worth at a time;
4. per spilled partition: reads the build rows back, builds a SingleNodeJoin, streams the partition's probe rows
   through it, and frees it; the next partition is read ahead meanwhile.
Only the join keys and the joined columns the probe pipeline reads are spilled; the pipeline still gets every
joined column, the unread ones as zero-stride stand-ins. P = SPILL_PARTITIONS; SPILL_RESIDENT_PARTITIONS caps k
(engines/pystachio/spill_partitions.csv; 0 spills everything). A partition whose build state would exceed
SPILL_PARTITION_BUDGET_BYTES stops the query with "spill partition too large" (no recursive repartitioning).
"""
import os
import time
from collections import deque

import torch
import torch.nn as nn

from helpers.flowcsd_observe import spill_stats
from helpers.key_filter import KeyFilter, filter_bytes_at_most, kernels
from helpers.spill_io import (WRITE_BATCHES_IN_FLIGHT, ChunkPrefetcher, SpillPartitions, gpu_bytes_available,
                              resident_partition_count, spill_partition_count)
from helpers.util import next_prime
from modules.single_node_join import SingleNodeJoin

# GPU memory kept free for the probe pipeline's growing state and allocator slack while resident partitions are
# chosen (probe_phase_reserve_bytes adds the probe side's batches). The largest such state among the spill queries
# is Q10's partial aggregates: 1.6 GB for the whole query at SF1000 (49M partial groups x 32 B), part of it during
# the probe phase.
PROBE_PIPELINE_RESERVE_BYTES = 2 << 30


def join_build_bytes(rows: int, row_bytes: int, hash_slots: int, tables: int) -> int:
    """Peak GPU bytes of SingleNodeJoin.build on one partition (also the sizing rule of spill_partitions.csv).

    Two copies of the build columns (the partition's morsels and the join's concatenated copy), four int64
    vectors per row (the hashes, their concatenation, the ones of the bincount, the arange of a table), the
    bincount and the "slot taken" mask (9 B per slot), and `tables` direct-mapped tables of (slots + 1) int64.
    """
    return 2 * rows * row_bytes + 4 * 8 * rows + 9 * hash_slots + tables * 8 * (hash_slots + 1)


class MorselList(nn.Module):
    """A child that hands out a list of morsels once (SingleNodeJoin.build concatenates them)."""

    def __init__(self, morsels: list[list[torch.Tensor]], dtypes: list[torch.dtype]):
        super().__init__()
        self.morsels = deque(morsels)
        self.dtypes = dtypes

    def forward(self):
        return self.morsels.popleft() if self.morsels else None

    def get_dtypes(self):
        return self.dtypes

    def get_morsel_count(self):
        return len(self.morsels)


class GraceHashJoin(nn.Module):
    def __init__(self, gpu, hash_table_init, left_child, left_index, right_child, right_index, store,
                 store_prefix="", project_build_side=None, *, build_spill_dirs, probe_spill_dirs, needed_columns):
        """Arguments as ShuffleWithJoin_MultiStream; store and store_prefix are unused (one GPU, nothing to exchange).

        build_spill_dirs / probe_spill_dirs: per-run spill directories of the two sides (':'-separated to stripe
        over several SSDs; run_query.sh creates them). needed_columns: positions in the joined layout (build
        columns after project_build_side, then probe columns) that probe_pipeline reads.
        """
        super().__init__()
        self._gpu = gpu
        self.left_child = left_child
        self.right_child = right_child
        self.partitions = spill_partition_count()
        self.resident_limit = resident_partition_count(self.partitions)  # forward() chooses k, at most this
        self.resident = 0
        self.budget_bytes = int(os.environ["SPILL_PARTITION_BUDGET_BYTES"])
        self.hash_slots = next_prime(hash_table_init // self.partitions)
        self._left_dtypes = left_child.get_dtypes()
        self._right_dtypes = right_child.get_dtypes()
        # Joined layout: build columns (Q5 keeps only project_build_side, join key first) then probe columns.
        left_columns = project_build_side if project_build_side is not None else list(range(len(self._left_dtypes)))
        left_count = len(left_columns)
        build_key = left_columns.index(left_index)
        # Spill only the keys and the joined columns the probe pipeline reads.
        build_keep = sorted({column for column in needed_columns if column < left_count} | {build_key})
        probe_keep = sorted({column - left_count for column in needed_columns if column >= left_count} | {right_index})
        self.build_source = [left_columns[position] for position in build_keep]  # child column of each kept one
        self.probe_source = probe_keep
        self.build_key_position = build_keep.index(build_key)
        self.probe_key_position = probe_keep.index(right_index)
        self.joined_positions = build_keep + [left_count + column for column in probe_keep]
        joined_dtypes = [self._left_dtypes[column] for column in left_columns] + list(self._right_dtypes)
        self.stand_ins = [torch.zeros((), dtype=dtype, device=gpu) for dtype in joined_dtypes]
        self.build_dtypes = [self._left_dtypes[column] for column in self.build_source]
        probe_dtypes = [self._right_dtypes[column] for column in self.probe_source]
        self.build_spill = SpillPartitions(build_spill_dirs, "build", self.partitions, self.resident_limit,
                                           self.build_dtypes, self.build_key_position)
        self.probe_spill = SpillPartitions(probe_spill_dirs, "probe", self.partitions, self.resident_limit,
                                           probe_dtypes, self.probe_key_position)
        self.build_spill.spilled_keys = []  # the keys of every spilled build row, until the filter is built
        # Modules nothing reads once the build side is drained (the query file adds its customer join): forward()
        # calls their release() before it chooses k, so their GPU memory goes to resident partitions.
        self.release_after_build_side = []
        self.key_filter = None
        kernels()  # compile the filter's CUDA kernels now, before the query runs
        self.build_side_tuples = 0
        self.probe_side_tuples = 0
        self.matched_pairs = 0

    def name(self):
        return "GraceHashJoin"

    def get_stats(self):
        return self.build_side_tuples, self.probe_side_tuples, self.matched_pairs

    def get_shuffled_bytes(self):
        """Bytes of the rows that went through the join per side (the shuffle join's definition: all child columns)."""
        left = self.build_side_tuples * sum(dtype.itemsize for dtype in self._left_dtypes)
        right = self.probe_side_tuples * sum(dtype.itemsize for dtype in self._right_dtypes)
        return left, right

    def forward(self, probe_pipeline=None):
        """Partition both sides (resident or spilled), then join the spilled partitions; the shuffle join's result dict."""
        if probe_pipeline is None:
            probe_pipeline = lambda *args, **kwargs: None
        start_ns = time.perf_counter_ns()
        with torch.cuda.nvtx.range("grace_partition_build"):
            self.build_side_tuples = self.spill_side(self.left_child, self.build_source, self.build_spill)
            self.build_spill.flush()  # the last batch; its writes finish while the probe side is partitioned
        spill_stats.mark("spill_build_partitioned_ns")
        for module in self.release_after_build_side:
            module.release()
        with torch.cuda.nvtx.range("grace_build_resident"):
            resident_joins = self.build_resident_joins()
            self.resident = len(resident_joins)
            self.build_spill.lower_resident(self.resident)  # writes the kept build rows of partitions k..P-1
            self.probe_spill.lower_resident(self.resident)
        spill_stats.set("spill_resident_partitions", self.resident)
        with torch.cuda.nvtx.range("grace_build_filter"):
            self.build_key_filter()
        # Probe rows of resident partitions are joined as soon as their batch is partitioned.
        self.probe_spill.use_resident_rows = lambda partition, columns: self.probe(
            resident_joins[partition], columns, probe_pipeline)
        with torch.cuda.nvtx.range("grace_partition_probe"):
            self.probe_side_tuples = self.spill_side(self.right_child, self.probe_source, self.probe_spill)
            self.probe_spill.flush()
            self.probe_spill.write_held()  # the last held rows of the filtered spilled partitions
        spill_stats.mark("spill_input_partitioned_ns")
        self.matched_pairs = sum(join.num_join_tuples for join in resident_joins if join is not None)
        self.probe_spill.use_resident_rows = None
        # Free the resident hash tables before the spilled partitions are built. Their probes ran on this stream:
        # wait for them, or the pool could hand the memory to a reader's decode while they still read it.
        torch.cuda.current_stream().synchronize()
        del resident_joins
        print(f"# Grace join: {self.partitions} partitions ({self.resident} resident, at most {self.resident_limit}), "
              f"{self.hash_slots} hash slots each, build/probe tuples {self.build_side_tuples}/"
              f"{self.probe_side_tuples}; {self.filter_summary()}", flush=True)
        # Every chunk must be on the SSD before it is read back; until here the writes overlapped the probe side.
        self.build_spill.finish_writes()
        self.probe_spill.finish_writes()
        self.join_spilled_partitions(probe_pipeline)
        self.build_spill.close()
        self.probe_spill.close()
        torch.cuda.synchronize(device=self._gpu)
        spill_stats.mark("spill_join_end_ns")
        return {"local_result": self.matched_pairs, "time_ns": time.perf_counter_ns() - start_ns,
                "shuffled_bytes": self.get_shuffled_bytes(), "stats": self.get_stats()}

    def build_resident_joins(self) -> list:
        """The joins of partitions 0..k-1, built here: k is the number of partitions, from 0 on, whose build peak
        (exact rows and hash tables) fits in the GPU memory still available when they are built, next to the probe
        phase's reserve and the key filter over the build rows of the partitions after them; at most
        SPILL_RESIDENT_PARTITIONS."""
        reserve_bytes = self.probe_phase_reserve_bytes()
        rows_after = self.build_side_tuples  # build rows of the partitions after the ones checked so far
        joins = []
        for partition in range(self.resident_limit):
            morsels = self.build_spill.resident_morsels[partition]
            rows_after -= sum(morsel[0].shape[0] for morsel in morsels)
            # The filter is built once k is chosen and kept during the probe phase: if this partition stays resident,
            # it covers the build rows of the partitions after it. While it is built, its transient key copies (at
            # most KEYS_PER_STEP int64 keys) fit in the probe-phase reserve, which is not in use yet.
            needed_bytes = self.build_bytes(partition, morsels) + filter_bytes_at_most(rows_after)
            if needed_bytes + reserve_bytes > gpu_bytes_available():
                break
            joins.append(self.build_join(self.build_spill.take_resident(partition)))
        return joins

    def probe_phase_reserve_bytes(self) -> int:
        """GPU memory the probe phase needs besides the resident joins: the probe side's batches (those being
        written, or held to be written when the key filter is on, the one being grouped and its reordered copy) and
        PROBE_PIPELINE_RESERVE_BYTES."""
        return (WRITE_BATCHES_IN_FLIGHT + 2) * self.probe_spill.flush_bytes + PROBE_PIPELINE_RESERVE_BYTES

    def build_key_filter(self) -> None:
        """The filter over the build keys of the spilled partitions k..P-1 (helpers/key_filter.py), which the probe
        side's partition step applies to their rows; none when every partition is resident (k = P), and then no probe
        row is tested. Frees the kept key columns."""
        spilled_keys = self.build_spill.spilled_keys
        self.build_spill.spilled_keys = None
        for name in ("build_ns", "build_keys", "false_positives_per_million", "apply_ns", "probe_rows_before",
                     "probe_rows_after"):
            spill_stats.set(f"spill_filter_{name}", 0)
        if self.resident == self.partitions:
            return
        start_ns = time.perf_counter_ns()
        self.key_filter = KeyFilter(spilled_keys)
        torch.cuda.current_stream().synchronize()
        spill_stats.set("spill_filter_build_ns", time.perf_counter_ns() - start_ns)
        spill_stats.set(f"spill_filter_{self.key_filter.kind}_bytes", self.key_filter.size_bytes)
        spill_stats.set("spill_filter_build_keys", self.key_filter.build_keys)
        spill_stats.set("spill_filter_false_positives_per_million", round(self.key_filter.false_positive_rate * 1e6))
        self.probe_spill.key_filter = self.key_filter

    def filter_summary(self) -> str:
        """The key filter's part of the "# Grace join:" line."""
        if self.key_filter is None:
            return "no key filter (every partition resident)"
        return (f"{self.key_filter.kind} filter {self.key_filter.size_bytes} B over {self.key_filter.build_keys} "
                f"build keys of spilled partitions, false-positive rate {self.key_filter.false_positive_rate:.4f}, "
                f"probe rows of spilled partitions {spill_stats.fields['spill_filter_probe_rows_before']} -> "
                f"{spill_stats.fields['spill_filter_probe_rows_after']}")

    def spill_side(self, child, source_columns: list[int], spill: SpillPartitions) -> int:
        """Partition all of one join side (only source_columns of every morsel); returns its row count.

        Calls the child exactly get_morsel_count() times, like the shuffle join, then once more to make sure no
        morsel is left: a child that under-reported its count would otherwise lose rows silently.
        """
        rows = 0
        for _ in range(child.get_morsel_count()):
            data = child()
            if data is None or data[0].shape[0] == 0:
                continue
            columns = [data[index] for index in source_columns]
            rows += columns[0].shape[0]
            spill.append(columns)
        if child() is not None:
            raise RuntimeError(f"{type(child).__name__} returned more morsels than its get_morsel_count()")
        return rows

    def join_spilled_partitions(self, probe_pipeline) -> None:
        """Join every spilled partition with rows on both sides, reading the next ones ahead meanwhile."""
        partitions = [partition for partition in range(self.resident, self.partitions)
                      if self.build_spill.partition_rows(partition) > 0 and self.probe_spill.partition_rows(partition) > 0]
        reads = [(spill, partition) for partition in partitions for spill in (self.build_spill, self.probe_spill)]
        prefetcher = ChunkPrefetcher(reads)
        for partition in partitions:
            with torch.cuda.nvtx.range(f"grace_join p={partition}"):
                build_morsels = [prefetcher.take() for _ in self.build_spill.chunks[partition]]
                self.build_bytes(partition, build_morsels)  # stops the query if the partition is too large
                join = self.build_join(build_morsels)
                del build_morsels  # the join keeps its own concatenated copy
                for _ in self.probe_spill.chunks[partition]:
                    self.probe(join, prefetcher.take(), probe_pipeline)
                self.matched_pairs += join.num_join_tuples
                # Free this partition's hash tables before the next one is built. The probes ran on this stream:
                # wait for them, or the pool could hand the memory to a reader's decode while they still read it.
                torch.cuda.current_stream().synchronize()
                del join

    def build_join(self, morsels: list[list[torch.Tensor]]):
        """A SingleNodeJoin on one partition's build morsels; None without build rows."""
        if not morsels:
            return None
        build_child = MorselList(morsels, self.build_dtypes)
        join = SingleNodeJoin(self._gpu, self.hash_slots, build_child)
        join.build(build_child, self.build_key_position)
        join.finish()
        return join

    def probe(self, join, probe_columns: list[torch.Tensor], probe_pipeline) -> None:
        """Probe a partition's join with some of its probe rows; the pipeline gets the full joined layout."""
        if join is None:
            return  # a resident partition without build rows: no matches
        for data in join.probe(probe_columns, self.probe_key_position):
            joined = [stand_in.expand(data[0].shape[0]) for stand_in in self.stand_ins]  # unread columns
            for position, column in zip(self.joined_positions, data):
                joined[position] = column
            probe_pipeline(joined)

    def build_bytes(self, partition: int, morsels: list[list[torch.Tensor]]) -> int:
        """Peak GPU bytes of building one partition (join_build_bytes with its exact rows and hash tables; 0 without
        rows). Stops the query with "spill partition too large" above SPILL_PARTITION_BUDGET_BYTES."""
        if not morsels:
            return 0
        keys = torch.cat([morsel[self.build_key_position] for morsel in morsels])
        rows = keys.shape[0]
        # SingleNodeJoin builds one table per row that shares a hash slot: count them as it does.
        tables = int(torch.bincount(torch.remainder(keys, self.hash_slots)).max().item())
        needed_bytes = join_build_bytes(rows, sum(dtype.itemsize for dtype in self.build_dtypes), self.hash_slots,
                                        tables)
        if needed_bytes > self.budget_bytes:
            raise RuntimeError(f"spill partition too large: build partition {partition} of {self.partitions} has "
                               f"{rows} rows and {tables} hash tables of {self.hash_slots} slots, about "
                               f"{needed_bytes / 2**30:.2f} GiB > SPILL_PARTITION_BUDGET_BYTES "
                               f"{self.budget_bytes / 2**30:.2f} GiB; give the query more partitions in "
                               f"spill_partitions.csv")
        return needed_bytes
