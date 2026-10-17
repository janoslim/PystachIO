"""Hybrid group-by with SSD spill partitions: Q18's pre-aggregation in the spill mode (SHUFFLE_TYPE=GRACE, patch 0003).

Same interface as AggregationPostCombine (forward per morsel, finalize, get_results, reset). forward() aggregates
each morsel as AggregationPostCombine does and splits the partial groups by the first group column % P
(helpers/spill_io.py): partitions 0..k-1 keep their partials in GPU memory, the others go to the data SSDs.
finalize() combines one partition at a time (all partial groups of one key are in the same partition), resident
partitions first, the spilled ones read a window ahead, and keeps only the groups that pass post_filter, so the
complete set of groups (1.5e9 orders at SF1000) never exists at once. Q18 passes its HAVING sum(l_quantity) > 300.
Handles one result column per group (Q18's sum). P = SPILL_PARTITIONS, k = SPILL_RESIDENT_PARTITIONS: fixed per
query, because the resident partials grow during the whole scan (the join chooses its k at run time instead).
"""
import os

import torch
import torch.nn as nn

from helpers.flowcsd_observe import spill_stats
from helpers.spill_io import ChunkPrefetcher, SpillPartitions, resident_partition_count, spill_partition_count
from modules.aggregation_module import group_by

# Peak GPU bytes per partial group in finalize's group_by + combine for an int64 group key and an int64 result
# (sizing rule of spill_partitions.csv): the partition's morsels and their concatenation (32 B), the row order,
# argsort output and sort scratch (about 24 B), the sorted key column and reordered row order (16 B), the sorted
# copies of both columns (16 B) and the inverse index of unique_consecutive (8 B).
AGGREGATION_BYTES_PER_PARTIAL_GROUP = 96


class SpillingAggregation(nn.Module):
    def __init__(self, agg_fun, combine_agg, gb_col_idxs, *, spill_dirs, post_filter, partial_dtypes,
                 projection=None):
        """agg_fun, combine_agg, gb_col_idxs, projection: as AggregationPostCombine.

        spill_dirs: per-run spill directory on a data SSD without the input table, or several separated by ':'
        (run_query.sh creates them). post_filter(aggs) -> bool mask of the final groups to keep (Q18: the HAVING).
        partial_dtypes: dtypes of a partial group's columns (the group columns, then the result).
        """
        super().__init__()
        self.agg_fun = agg_fun
        self.combine_agg = combine_agg
        self.gb_col_idxs = gb_col_idxs
        self.projection = projection
        self.post_filter = post_filter
        self.partitions = spill_partition_count()
        self.resident = resident_partition_count(self.partitions)
        self.budget_bytes = int(os.environ["SPILL_PARTITION_BUDGET_BYTES"])
        # Created here, before the query runs, like the join's: the spill files and the I/O threads' staging buffers.
        self.spill = SpillPartitions(spill_dirs, "aggregation", self.partitions, self.resident, partial_dtypes,
                                     key_index=0)
        spill_stats.set("spill_resident_partitions", self.resident)
        self.reset()

    def reset(self):
        self.final_groups = None
        self.final_aggs = None

    def get_results(self):
        return (self.final_groups, self.final_aggs)

    def forward(self, data):
        """Aggregate one morsel and partition its partial groups (the group columns, then the result)."""
        sorted_data, groups_unique, group_idx = group_by(data, self.gb_col_idxs, projection=self.projection)
        with torch.cuda.nvtx.range("Agg fun"):
            results = self.agg_fun(sorted_data, groups_unique, group_idx)
        columns = [groups_unique[:, index].contiguous() for index in range(groups_unique.shape[1])] + [results]
        assert [column.dtype for column in columns] == self.spill.dtypes, "partial groups differ from partial_dtypes"
        self.spill.append(columns)

    def finalize(self, drop_partial=False):
        """Combine the partial groups partition by partition and keep the groups that pass post_filter.

        drop_partial is accepted for AggregationPostCombine's interface; the partials are always dropped.
        """
        self.spill.flush()
        spill_stats.mark("spill_input_partitioned_ns")
        kept_groups = []
        kept_aggs = []
        group_count = 0
        for partition in range(self.resident):
            morsels = self.spill.take_resident(partition)
            if morsels:
                group_count += self.combine_partition(partition, morsels, kept_groups, kept_aggs)
        # Every chunk must be on the SSD before it is read back; the resident partitions overlapped the last writes.
        self.spill.finish_writes()
        spilled = [partition for partition in range(self.resident, self.partitions)
                   if self.spill.partition_rows(partition) > 0]
        prefetcher = ChunkPrefetcher([(self.spill, partition) for partition in spilled])
        for partition in spilled:
            morsels = [prefetcher.take() for _ in self.spill.chunks[partition]]
            group_count += self.combine_partition(partition, morsels, kept_groups, kept_aggs)
        self.spill.close()
        self.final_groups = torch.cat(kept_groups)
        self.final_aggs = torch.cat(kept_aggs)
        torch.cuda.current_stream().synchronize()
        spill_stats.mark("spill_join_end_ns")
        # Every order is one group before the HAVING: pystachio_result.py checks this count (1.5e6 x SF for Q18).
        print(f"# SpillingAggregation: {self.partitions} partitions ({self.resident} resident), {group_count} groups "
              f"before post_filter, {self.final_groups.shape[0]} after", flush=True)

    def combine_partition(self, partition: int, morsels: list[list[torch.Tensor]], kept_groups: list,
                          kept_aggs: list) -> int:
        """Combine one partition's partial groups and keep the final groups that pass post_filter; returns the
        number of groups before post_filter."""
        with torch.cuda.nvtx.range(f"aggregation_finalize p={partition}"):
            self.check_partition_fits(partition, sum(morsel[0].shape[0] for morsel in morsels))
            columns = [torch.cat([morsel[index] for morsel in morsels]) for index in range(len(morsels[0]))]
            groups = torch.stack(columns[:-1], dim=1)
            sorted_data, groups_unique, group_idx = group_by([groups, columns[-1]], [], preformat=True)
            aggs = self.combine_agg(sorted_data, groups_unique, group_idx)
            keep = self.post_filter(aggs)
            kept_groups.append(groups_unique[keep])
            kept_aggs.append(aggs[keep])
            return groups_unique.shape[0]

    def check_partition_fits(self, partition: int, rows: int) -> None:
        """Stop with "spill partition too large" if combining this partition would exceed the per-partition budget."""
        needed_bytes = rows * AGGREGATION_BYTES_PER_PARTIAL_GROUP
        if needed_bytes > self.budget_bytes:
            raise RuntimeError(f"spill partition too large: aggregation partition {partition} of {self.partitions} "
                               f"has {rows} partial groups, about {needed_bytes / 2**30:.2f} GiB > "
                               f"SPILL_PARTITION_BUDGET_BYTES {self.budget_bytes / 2**30:.2f} GiB; give the query "
                               f"more partitions in spill_partitions.csv")
