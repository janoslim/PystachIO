# KVIKIO_COMPAT_MODE=ON python helpers/Parquet_FIFO.py helpers/userdata.parquet 1 userdata id,salary --verbose
# python helpers/Parquet_FIFO.py /raid/jluo/lineitem_sf300/temp_plain/comp-none_rowgroup-10000000_pagerow-100000_dict-false_stats-chunk.parquet 32 lineitem l_partkey,l_orderkey,l_extendedprice --verbose
# STORAGE_LOGGING_LEVEL=DEBUG python helpers/Parquet_FIFO.py /raid/jluo/parquet-rewriting/Aug13duckdb/tpch-sf1/orders.parquet 1 orders o_orderdate
from helpers.FIFO import FIFO, TensorMorsel # FIX THIS IF YOU RUN ABOVE
# from FIFO import FIFO, TensorMorsel # FIX THIS IF YOU RUN ABOVE

import sys
import time
import torch
from torch import distributed as dist
import pylibcudf as plc
import rmm
from rmm.pylibrmm.stream import Stream, DEFAULT_STREAM, PER_THREAD_DEFAULT_STREAM
import cupy
import threading
import pyarrow.parquet as pq
import itertools
import logging
import os
import humanize
import nvtx
from helpers.beemon import Beemon
from modules.filter_module import FilterCondition, FilterOp, TableFilter
from helpers.thread_pool_manager import (
    get_global_thread_pool, 
    shutdown_global_thread_pool, 
    join_one_reader, 
    add_thread_future, 
    join_all_threads, 
    clear_thread_futures
)
from helpers.util import parquet_type_to_torch_dtype, to_torch_dtype
from helpers.flowcsd_observe import first_read_begin, first_read_end
from concurrent.futures import ThreadPoolExecutor
import cupy as cp
import numpy as np
import random

# Global cache for parquet metadata
CUDF_PARQUET_METADATA_CACHE = {}
PYARROW_PARQUET_METADATA_CACHE = {}

logger = logging.getLogger(__name__)
def logger_setup():
    storage_logging_level = os.environ.get("STORAGE_LOGGING_LEVEL", "INFO").upper()
    logging.basicConfig(level=storage_logging_level, format='%(asctime)s - %(levelname)s: %(message)s')
    logger.info(f"Logging level set to {storage_logging_level}")

def resolve_parquet_files(parquet_file: str) -> list[str]:
    """Resolve a single Parquet file, a directory of *.parquet part files, or a <table>.parquet path
    whose sibling directory <table>/ holds <table>-*.parquet part files."""
    if os.path.isdir(parquet_file):
        files = sorted(os.path.join(parquet_file, f) for f in os.listdir(parquet_file) if f.endswith(".parquet"))
        if files:
            return files
    stem_dir = parquet_file[:-8] if parquet_file.endswith(".parquet") else parquet_file
    if os.path.isdir(stem_dir):
        files = sorted(os.path.join(stem_dir, f) for f in os.listdir(stem_dir) if f.endswith(".parquet"))
        if files:
            return files
    if os.path.isfile(parquet_file):
        return [parquet_file]
    raise FileNotFoundError(f"No Parquet file(s) found for {parquet_file}")

class ParquetFIFOReader:
    def __init__(self, parquet_file: str, table_name: str, column_names: list[str], num_threads: int = 1,
                 world_rank: int = 0, world_size: int = 1, table_filter: TableFilter=None, next_reader=None, beemon: Beemon = None, full_read_mode: bool = False, prune_flag: bool = True):
        logger_setup() # INFO level by default
        self.parquet_file = parquet_file
        self.table_name = table_name
        self.column_names = column_names
        self.num_threads = num_threads
        self.world_rank = world_rank
        self.world_size = world_size
        self.table_filter = table_filter
        self.full_read_mode = full_read_mode # for table duplicated in all nodes
        self.prepare_time = None
        self.prune_flag = prune_flag
        
        # Reader chaining
        self.prev_reader = None
        self.next_reader = next_reader
        self.beemon = beemon
        self.start_time = None
        self.end_time = None

        ### Parquet metadata (single file or multi-part directory) ###
        # Emit NVTX metadata scope so pyarrow.parquet.read_metadata and row-group pruning
        # in ParquetFIFOReader.__init__ are attributed to Stage 5 (CPU Metadata) rather than Else.
        torch.cuda.nvtx.range_push("pystachio::metadata::read_and_prune")
        self.parquet_files = resolve_parquet_files(self.parquet_file)
        part_infos = []
        total_meta_ns = 0
        for part_file in self.parquet_files:
            if part_file in CUDF_PARQUET_METADATA_CACHE:
                logger.info(f"[ParquetFIFOReader] Using cached metadata for {part_file}")
                part_meta, part_meta_ns = CUDF_PARQUET_METADATA_CACHE[part_file]
            else:
                logger.info(f"[ParquetFIFOReader] Reading and caching metadata for {part_file}")
                tmpstart = time.perf_counter_ns()
                part_meta = plc.io.parquet_metadata.read_parquet_metadata(plc.io.SourceInfo([part_file]))
                tmpend = time.perf_counter_ns()
                part_meta_ns = tmpend - tmpstart
                CUDF_PARQUET_METADATA_CACHE[part_file] = (part_meta, part_meta_ns)
            total_meta_ns += part_meta_ns
            part_meta_ptr = part_meta.get_aggregate_reader_metadata_ptr()
            if part_file in PYARROW_PARQUET_METADATA_CACHE:
                logger.info(f"[ParquetFIFOReader] Using pyarrow cached metadata for {part_file}")
                part_pq_meta = PYARROW_PARQUET_METADATA_CACHE[part_file]
            else:
                logger.info(f"[ParquetFIFOReader] Reading and caching pyarrow metadata for {part_file}")
                part_pq_meta = pq.read_metadata(part_file)
                PYARROW_PARQUET_METADATA_CACHE[part_file] = part_pq_meta
            part_infos.append((part_file, part_meta_ptr, part_pq_meta))

        self.meta = CUDF_PARQUET_METADATA_CACHE[self.parquet_files[0]][0]
        CUDF_PARQUET_METADATA_CACHE[self.parquet_file] = (self.meta, total_meta_ns)
        self.metadata_ptr = part_infos[0][1]
        pq_metadata = part_infos[0][2]
        schema = pq_metadata.schema.to_arrow_schema()

        # Interleave row groups round-robin across part files so consecutive tasks cycle across SSDs.
        per_part_rgs = [
            [(part_file, part_meta_ptr, part_pq_meta, local_rg) for local_rg in range(part_pq_meta.num_row_groups)]
            for part_file, part_meta_ptr, part_pq_meta in part_infos
        ]
        self.rg_sources = [
            rg_entry
            for group in itertools.zip_longest(*per_part_rgs)
            for rg_entry in group
            if rg_entry is not None
        ]
        
        prepare_timer_start = time.perf_counter_ns()
        logger.info(f"[ParquetFIFOReader] [Table {self.table_name}] has #RG: {len(self.rg_sources)} across {len(self.parquet_files)} file(s)")
        if self.full_read_mode:
            # no distributed partitioning, read all RG on current rank
            logger.info(f"[ParquetFIFOReader] [Table {self.table_name}] Full read mode enabled, skipping distributed partitioning")
            pruned_rowgroups = self.prune_rowgroups(pq_metadata, schema)
            logger.info(f"[ParquetFIFOReader] [Table {self.table_name}] Pruned {len(self.rg_sources) - len(pruned_rowgroups)} row groups using {"empty-" if not self.table_filter else ""}filters, remaining: {len(pruned_rowgroups)}")
            self.rank_rowgroups = pruned_rowgroups
            self.rank_num_rowgroups = len(self.rank_rowgroups)
        else:
            # Pruning in master-node and broadcast to other nodes
            if self.world_rank == 0:
                pruned_rowgroups = self.prune_rowgroups(pq_metadata, schema)
                logger.info(f"[ParquetFIFOReader] [Table {self.table_name}] [Master] Pruned {len(self.rg_sources) - len(pruned_rowgroups)} row groups using {"empty-" if not self.table_filter else ""}filters, remaining: {len(pruned_rowgroups)}")
                
                self.all_rank_partitions = {}
                for rank in range(self.world_size):
                    rank_rowgroups = self.partition_RG_to_ranks(pruned_rowgroups, rank)
                    self.all_rank_partitions[rank] = {'rank_rowgroups': rank_rowgroups} # maybe I will add more
                logger.info(f"[ParquetFIFOReader] [Table {self.table_name}] [Master] Calculated partitioning for all {self.world_size} ranks")
            else:
                self.all_rank_partitions = {}
            self.sync_partition_results() # Broadcast partition results from rank 0 to all other ranks
            ### Partitioning into nodes ###
            partition_data = self.all_rank_partitions[self.world_rank]
            self.rank_rowgroups = partition_data['rank_rowgroups']
            if len(self.parquet_files) == 1:
                random.shuffle(self.rank_rowgroups)
            self.rank_num_rowgroups = len(self.rank_rowgroups)
        logger.info(f"[ParquetFIFOReader] [Table {self.table_name}] Rank {self.world_rank} will create {self.rank_num_rowgroups} futures (one per RG) for dynamic work distribution: {self.rank_rowgroups}")
        torch.cuda.nvtx.range_pop()
        self.thread_pool = get_global_thread_pool(self.num_threads)
        prepare_timer_end = time.perf_counter_ns()
        self.prepare_time = prepare_timer_end - prepare_timer_start
        self.stream_pool = [torch.cuda.Stream(device="cuda:0") for _ in range(self.num_threads)] # TODO: (jigao) need also to be a global one
        logger.info(f"[ParquetFIFOReader] [Table {self.table_name}] Prepare Time (pruning, syncing, logging, partitioning, thread-pool, stream-pool): {(prepare_timer_end - prepare_timer_start) / 1_000_000:.2f} ms")
        
        ### Decimal & StorageSize 
        torch.cuda.nvtx.range_push("pystachio::metadata::column_stats")
        self.column_schema = {}
        self.is_decimal = {}
        self.storage_bytes = {} # col_name -> [actual bytes, uncompressed bytes, after filter bytes, file actual bytes, file uncompressed bytes] 
        self.dtypes = []
        # https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ColumnSchema.html
        for col_name in self.column_names:
            col_idx = schema.get_field_index(col_name)
            column_meta = pq_metadata.schema.column(col_idx)
            self.column_schema[col_name] = column_meta
            self.is_decimal[col_name] = column_meta.logical_type.type.startswith("DECIMAL")
            self.dtypes.append(parquet_type_to_torch_dtype(column_meta.physical_type))
            # Get storage bytes and uncompressed size for this column
            col_storage_bytes = 0
            col_uncompressed_bytes = 0
            col_file_actual_bytes = 0
            col_file_uncompressed_bytes = 0
            # Calculate storage for rank rowgroups
            for rg_idx, rg in enumerate(self.rank_rowgroups):
                _, _, part_pq_meta, local_rg = self.rg_sources[rg]
                row_group = part_pq_meta.row_group(local_rg)
                rg_column = row_group.column(col_idx)
                assert rg_column.path_in_schema == col_name
                col_storage_bytes += rg_column.total_compressed_size
                col_uncompressed_bytes += self.dtypes[-1].itemsize * row_group.num_rows
            # Calculate storage for entire table across all part files
            for _, _, part_pq_meta, local_rg in self.rg_sources:
                row_group = part_pq_meta.row_group(local_rg)
                rg_column = row_group.column(col_idx)
                assert rg_column.path_in_schema == col_name
                col_file_actual_bytes += rg_column.total_compressed_size
                col_file_uncompressed_bytes += self.dtypes[-1].itemsize * row_group.num_rows
            self.storage_bytes[col_name] = [col_storage_bytes, col_uncompressed_bytes, 0, col_file_actual_bytes, col_file_uncompressed_bytes] # 0 in filtered bytes we can know yet
        
        torch.cuda.nvtx.range_pop()
        logger.info(f"[ParquetFIFOReader] table={self.table_name} file={self.parquet_file} parts={len(self.parquet_files)}")
        logger.info(f"                    [GLOBAL] total_rowgroups={len(self.rg_sources)} threads={self.num_threads} columns={self.column_names}")
        logger.info(f"                    world_rank={self.world_rank}, world_size={self.world_size}")
        logger.info(f"                    rank_num_rowgroups={self.rank_num_rowgroups} (RG IDs: {self.rank_rowgroups})")
        logger.info(f"                    Physical types: {' | '.join([f'{col}:{meta.physical_type}' for col, meta in self.column_schema.items()])}") 
        logger.info(f"                    dtypes={self.dtypes}")
        logger.info(f"                    Logical types: {' | '.join([f'{col}:{meta.logical_type.type} with {meta.logical_type}'  for col, meta in self.column_schema.items()])}") 
        logger.info(f"                    Is Decimal: {' | '.join([f'{col}:{self.is_decimal[col]}' for col in self.column_names ])}") 
        logger.info(f"                    Storage Bytes: {' | '.join([f'{col}:{humanize.naturalsize(self.storage_bytes[col][0], binary=True)}(actual):{humanize.naturalsize(self.storage_bytes[col][1], binary=True)}(uncompressed)' for col in self.column_names ])}") 
        logger.info(f"                    Total Storage Bytes: {humanize.naturalsize(sum(v[0] for v in self.storage_bytes.values()), binary=True)}(actual):{humanize.naturalsize(sum(v[1] for v in self.storage_bytes.values()), binary=True)}(uncompressed)") 
        logger.info(f"[ParquetFIFOReader] [Table {self.table_name}] Dynamic work distribution: {self.rank_num_rowgroups} RG futures will be submitted to thread pool with {self.num_threads} max threads")

        # FIFO now uses number of rowgroups as producer count instead of thread count
        queue_max_percentage = os.environ.get("FIFO_QUEUE_MAX_PERCENTAGE", "100")
        logger.info(f"FIFO_QUEUE_MAX_PERCENTAGE: {queue_max_percentage}")
        if self.rank_num_rowgroups <= 20:
            queue_max_percentage = 100 # skip the max-setup for the table with small number of RGs
        fifo_maxsize = int(self.rank_num_rowgroups * (float(queue_max_percentage) / 100))
        if fifo_maxsize < 1:
            fifo_maxsize = 1
        logger.info(f" -> max size to set: {fifo_maxsize}")
        self.fifo = FIFO(
            parent=self,
            num_producers=self.rank_num_rowgroups, 
            table_name=self.table_name, 
            columns=self.column_names, 
            maxsize=fifo_maxsize
        )
        self.producer_futures = [] # Store futures for each RG task
        
        # Thread-safe counters for filtered bytes
        self._filtered_bytes_lock = threading.Lock()
        self._filtered_bytes = {col_name: 0 for col_name in self.column_names}
        
        self._started = threading.Event() # atomic boolean
        self._finished = False
    
    def prune_rowgroups(self, pq_metadata, schema) -> list[int]:
        if not self.prune_flag:
            logger.info(f"[ParquetFIFOReader] [Table {self.table_name}] Pruning disabled, skipping pruning")
            return list(range(len(self.rg_sources)))
        if self.table_filter is None:
            logger.info(f"[ParquetFIFOReader] [Table {self.table_name}] No filters provided, skipping pruning")
            return list(range(len(self.rg_sources)))
        remaining_rgs = [] # the returned list[int]
        for rg_idx, (_, _, part_pq_meta, local_rg) in enumerate(self.rg_sources):
            row_group = part_pq_meta.row_group(local_rg)
            should_include = True
            for column_idx, filter_list in self.table_filter.column_conditions.items():
                # the column_idx is local in the reading list, not the index in the full table
                assert column_idx < len(self.column_names)
                col_name = self.column_names[column_idx]
                col_schema_idx = schema.get_field_index(col_name)
                
                # Skip pruning for decimal columns DUE TO ISSUE174 & ArrowCPP
                column_meta = part_pq_meta.schema.column(col_schema_idx)
                if column_meta.logical_type.type.startswith("DECIMAL"):
                    logger.debug(f"[ParquetFIFOReader] [Issue174] Skipping pruning for decimal column {col_name}")
                    continue
                
                try:
                    rg_column = row_group.column(col_schema_idx)
                    if rg_column.statistics is None:
                        logger.debug(f"[ParquetFIFOReader] No statistics for RG {rg_idx}, column {col_name}, including by default")
                        continue
                    stats = rg_column.statistics
                    if not stats.has_min_max:
                        logger.debug(f"[ParquetFIFOReader] No min/max stats for RG {rg_idx}, column {col_name}, including by default")
                        continue
                    
                    min_val = stats.min.as_py() if hasattr(stats.min, 'as_py') else stats.min
                    max_val = stats.max.as_py() if hasattr(stats.max, 'as_py') else stats.max
                    assert min_val is not None and max_val is not None
                    # Check if this row group can be pruned based on the filters
                    # All filters for this column must be satisfied (AND logic)
                    column_satisfies_all_filters = True
                    for filter_obj in filter_list:
                        logger.debug(f"[ParquetFIFOReader] Evaluating RG {rg_idx} column={col_name} min={min_val} max={max_val} WITH filter {filter_obj.op} {filter_obj.value}")
                        if not filter_obj.evaluate_on_range(min_val, max_val):
                            column_satisfies_all_filters = False
                            logger.debug(f"[ParquetFIFOReader] Pruning RG {rg_idx}: column {col_name} filter {filter_obj} cannot be satisfied with range [{min_val}, {max_val}]")
                            break
                    if not column_satisfies_all_filters:
                        should_include = False
                        break
                    else:
                        logger.debug(f"[ParquetFIFOReader] Including RG {rg_idx}: column {col_name} all filters may be satisfied with range [{min_val}, {max_val}]")
                except Exception as e:
                    logger.warning(f"[ParquetFIFOReader] Error processing statistics for RG {rg_idx}, column {col_name}: {e}")
                    continue
            if should_include:
                remaining_rgs.append(rg_idx)
        if not self.full_read_mode and len(remaining_rgs) < self.world_size:
            logger.debug(f"[ParquetFIFOReader] [Table {self.table_name}] After pruning, only {len(remaining_rgs)} row groups remain < world_size={self.world_size}, leading to some nodes having no data to process.")
        return remaining_rgs

    def partition_RG_to_ranks(self, pruned_rowgroups: list[int], world_rank: int) -> list[int]:
        total_rowgroups = len(pruned_rowgroups)
        rgs_per_rank = total_rowgroups // self.world_size
        remainder = total_rowgroups % self.world_size
        logger.debug(f"[ParquetFIFOReader] [Table {self.table_name}] Rank {world_rank}: total_rowgroups={total_rowgroups}, world_size={self.world_size}")
        logger.debug(f"[ParquetFIFOReader] [Table {self.table_name}] Rank {world_rank}: rgs_per_rank={rgs_per_rank}, remainder={remainder}")
        rank_start_idx = world_rank * rgs_per_rank + min(world_rank, remainder)
        if world_rank < remainder:
            rank_end_idx = rank_start_idx + rgs_per_rank + 1
        else:
            rank_end_idx = rank_start_idx + rgs_per_rank
        rank_num_rowgroups = rank_end_idx - rank_start_idx
        rank_rowgroups = pruned_rowgroups[rank_start_idx:rank_end_idx]
        logger.info(f"[ParquetFIFOReader] [Table {self.table_name}] [partition_RG_to_ranks] Rank {world_rank}: {rank_num_rowgroups} row groups {rank_rowgroups}")
        return rank_rowgroups

    def sync_partition_results(self):
        if self.world_size == 1:
            logger.info(f"[ParquetFIFOReader] [Table {self.table_name}] Single Node, no partition synchronization needed")
            return
        try:
            if self.world_rank == 0:
                # Master node: broadcast the partition results as a dictionary
                broadcast_data = [self.all_rank_partitions]
                dist.broadcast_object_list(broadcast_data, src=0)
                logger.info(f"[ParquetFIFOReader] [Table {self.table_name}] [Master] Broadcasted partition results to all ranks")
            else:
                # Worker nodes: receive the partition results
                broadcast_data = [None]
                dist.broadcast_object_list(broadcast_data, src=0)
                self.all_rank_partitions = broadcast_data[0]
                logger.info(f"[ParquetFIFOReader] [Table {self.table_name}] [Worker {self.world_rank}] Received partition results from master")
            dist.barrier()
        except Exception as e:
            logger.error(f"[ParquetFIFOReader] [Table {self.table_name}] Error during partition result synchronization: {e}")
            sys.exit(1)

    def set_next_reader(self, next_reader):
        assert isinstance(next_reader, ParquetFIFOReader), "next_reader must be a ParquetFIFOReader"
        self.next_reader = next_reader
        self.next_reader.prev_reader = self
        return

    def get_dtypes(self):
        return self.dtypes

    def get_morsel_count(self) -> int:
        return self.rank_num_rowgroups # the local RGs to handle in this world rank
        
    def get_total_num_rowgroups(self) -> int:
        return len(self.rg_sources) # the total&global number of RGs across all part files

    @torch.inference_mode()
    def thread_read_producer(self, rg_idx: int):
        """Producer function: read a single row group and enqueue tensors."""
        thread_idx = int(threading.current_thread().name.replace("th_","")) # check get_global_thread_pool with prefix th_
        if __debug__:
            logger.debug(f"Thread name {threading.current_thread().name} & id {thread_idx}: starting RG_IDX={rg_idx} for table {self.table_name}")
        torch_stream = self.stream_pool[thread_idx]
        cupy_stream = cupy.cuda.ExternalStream(torch_stream.cuda_stream)
        th_stream = rmm.pylibrmm.stream.Stream(cupy_stream)
        filtered_bytes = {col_name: 0 for col_name in self.column_names}

        # Check if queue is full before reading, spin wait if needed
        while self.fifo._queue.full():
            time.sleep(0.001)
        
        try:
            torch.cuda.nvtx.range_push(f"RG_IDX={rg_idx} {self.table_name} RG={self.rank_rowgroups[rg_idx]}")
            rg = self.rank_rowgroups[rg_idx]
            part_file, metadata_ptr, _, local_rg = self.rg_sources[rg]
            options = plc.io.parquet.ParquetReaderOptions.builder(
                plc.io.SourceInfo([part_file])
            ).build()
            options.set_row_groups([[local_rg]])
            options.set_columns(self.column_names)
            options.set_aggregate_reader_metadata(metadata_ptr)
            first_read = first_read_begin() # FlowCSD observation: one-time libcudf initialization happens in the first call
            plc_table_w_meta = plc.io.parquet.read_parquet(options, th_stream)
            if first_read:
                first_read_end() # FlowCSD observation
            torch.cuda.nvtx.range_push(f"RG_IDX={rg_idx} {self.table_name} RG={rg} TBL")
            plc_table = plc_table_w_meta.tbl
            torch.cuda.nvtx.range_pop() 
            with torch.cuda.stream(torch_stream):
                filter_mask = torch.full((plc_table.columns()[0].size(),), True, dtype=torch.bool, device="cuda:0")

            # Prepare TensorMorsel
            torch.cuda.nvtx.range_push(f"RG_IDX={rg_idx} {self.table_name} RG={rg} PREPARE")
            tensors = []
            for col_idx, col_name in enumerate(self.column_names):
                torch.cuda.nvtx.range_push(f"RG_IDX={rg_idx} {self.table_name} RG={rg} {col_name} PREPARE")
                plc_column = plc_table.columns()[col_idx]
                pytorch_dtype = to_torch_dtype(plc_column.type().id())
                tensor = torch.as_tensor(plc_column.data(), device="cuda").view(pytorch_dtype)  # zero-copy
                
                torch.cuda.nvtx.range_pop()
                assert pytorch_dtype == parquet_type_to_torch_dtype(self.column_schema[col_name].physical_type), f"{plc_column.type().id().name} & {pytorch_dtype} & {parquet_type_to_torch_dtype(self.column_schema[col_name].physical_type)} dtype mismatch"
                torch.cuda.nvtx.range_push(f"RG_IDX={rg_idx} {self.table_name} RG={rg} {col_name} APPEND")
                # TODO: to preserve space when loading small domain size columns (int8, int16, etc.),
                #       columns could be converted here (after they have already been materialized in full ...)
                # if col_name == "l_returnflag" or col_name == "l_linestatus":
                #     with torch.cuda.stream(torch_stream):
                #         tensor = tensor.to(torch.int8)
                #     torch_stream.synchronize
                tensors.append(tensor)
                torch.cuda.nvtx.range_pop() 
                if __debug__:
                    logger.debug(f"[RG_IDX={rg_idx}] [Table {self.table_name}] [Column] RG{rg} col_name={col_name} col_size={plc_column.size()}, !datatype!={plc_column.type().id().name}\n"
                                 f"    [Tensor] RG{rg} col_name={col_name}: {tensor} {tensor.device} {tensor.shape} {tensor.dtype}")
                # conjunctive update filter mask
                torch.cuda.nvtx.range_push(f"RG_IDX={rg_idx} {self.table_name} RG={rg} {col_name} FILTER")
                if self.table_filter and col_idx in self.table_filter.column_conditions.keys():
                    assert self.table_filter.column_conditions[col_idx] is not None
                    with torch.cuda.stream(torch_stream):
                        filter_mask &= self.table_filter.column_lambdas[col_idx](tensor) # e.g. lambda col_tensor: torch.eq(col_tensor,10)
                        if __debug__:
                            logger.debug(f"[RG_IDX={rg_idx}] [Table {self.table_name}] Applying filter: {self.table_filter.column_conditions[col_idx]} with filter_mask: {filter_mask} {filter_mask.shape} {filter_mask.dtype}")
                torch.cuda.nvtx.range_pop() # FILTER

            # filter
            torch.cuda.nvtx.range_push(f"RG_IDX={rg_idx} {self.table_name} RG={rg} FILTER FINAL")
            # TODO: (jigao) this take 10% time of reading... optimize it
            if self.table_filter:
                with torch.cuda.stream(torch_stream):
                    for (cola,colb,row_lambda) in self.table_filter.row_lambdas:
                        filter_mask &= row_lambda(tensors[cola], tensors[colb])
                    for tensoridx, tensor in enumerate(tensors):
                        # tensors[tensoridx] = torch.masked_select(tensor, filter_mask)
                        mask_as_indices = filter_mask.view(-1).nonzero().squeeze()
                        tensors[tensoridx] = torch.index_select(tensor, 0, mask_as_indices)
                        if __debug__:
                            logger.debug(f"[RG_IDX={rg_idx}] [Table {self.table_name}] After filtering RG{rg} col_name={self.column_names[tensoridx]}: {tensors[tensoridx]} {tensors[tensoridx].device} {tensors[tensoridx].shape} {tensors[tensoridx].dtype}")
                torch_stream.synchronize()
            torch.cuda.nvtx.range_pop() # FILTER FINAL

            ##################################
            if os.environ["TPCH_QUERY"] == "reader":
                return # for MULTI_SSD_READER skip the memory stall
            ##################################
            
            torch.cuda.nvtx.range_push(f"RG_IDX={rg_idx} {self.table_name} RG={rg} FIFO PUT")
            self.fifo.put(TensorMorsel(self.column_names, tensors))
            for col_name,tensor in zip(self.column_names, tensors):
                filtered_bytes[col_name] += tensor.numel()*tensor.element_size()
            if __debug__:
                logger.debug(f"[RG_IDX={rg_idx} END] [Table {self.table_name}] RG={rg} cols={self.column_names}")
            torch.cuda.nvtx.range_pop() # PUT 
            torch.cuda.nvtx.range_pop() # PREPARE
        except Exception as e:
            print(f"[RG_IDX={rg_idx} Error] [Table {self.table_name}] {e}", flush=True)
        finally:
            self.fifo.producer_finish()
            # Store filtered bytes metadata for this task (rowgroup)
            with self._filtered_bytes_lock:
                for col_name, bytes_count in filtered_bytes.items():
                    self._filtered_bytes[col_name] += bytes_count
            torch.cuda.nvtx.range_pop() # FOR THIS TASK
            if self.fifo.is_finished():
                self.end_time = time.perf_counter_ns()
                # self._finished = True ## set in the join()
                # Not to join or start next reader here, because it is a thread inside the thread pool

            if __debug__:
                logger.debug(f"[ParquetFIFOReader] [Table {self.table_name}] RG_IDX {rg_idx} finished reading actual RG {self.rank_rowgroups[rg_idx]}\n"
                             f"    is_finished={self.fifo.is_finished()}")

    def start(self):
        if self._started.is_set():
            return # could be started in the reader chain
        self.start_time = time.perf_counter_ns()
        # Submit one future per rowgroup for dynamic work distribution
        for rg_idx in range(self.rank_num_rowgroups):
            if __debug__:
                logger.debug(f"[ParquetFIFOReader] [Table {self.table_name}] Rank {self.world_rank} Submitting RG_idx={rg_idx} (actual RG={self.rank_rowgroups[rg_idx]}) as future")
            future = self.thread_pool.submit(self.thread_read_producer, rg_idx)
            self.producer_futures.append(future)
            # Add to global thread collection for batched joining
            add_thread_future(future, self, rg_idx)
        self._started.set()
        logger.info(f"[ParquetFIFOReader] [Table {self.table_name}] Rank {self.world_rank} Submitted {len(self.producer_futures)} RG futures for dynamic execution by {self.num_threads} max threads")
        if self.rank_num_rowgroups == 0:
            self._finished = True
            self.fifo._finished.set() # see FIFO producer_finish
            self.end_time = self.start_time
            logger.debug(f"[ParquetFIFOReader] [Table {self.table_name}] Nothing (no thread future) to read in this node. Marked as finished immediately.")
        if __debug__:
            logger.debug(f"[ParquetFIFOReader] [Table {self.table_name}] Rank {self.world_rank} Submitted {len(self.producer_futures)} RG futures for {len(self.rank_rowgroups)} row groups")
    
    def parallel_start(self):
        """
        Alternative to start(): creates own thread pool for independent parallel execution. Ignores reader chain order - all readers execute independently.
        """
        if self._started.is_set():
            return
        self.start_time = time.perf_counter_ns()
        # Create own thread pool instead of using global one
        self.thread_pool = ThreadPoolExecutor(max_workers=self.num_threads, thread_name_prefix=f"th")
        for rg_idx in range(self.rank_num_rowgroups):
            future = self.thread_pool.submit(self.thread_read_producer, rg_idx)
            self.producer_futures.append(future)
        self._started.set()
        logger.info(f"[ParquetFIFOReader] [ParallelStart] [Table {self.table_name}] Created own thread pool and submitted {len(self.producer_futures)} RG futures")

    @staticmethod
    def parallel_start_all(readers_list):
        """
        Static method to start all readers in parallel mode with own thread pools.
        """
        for reader in readers_list:
            reader.parallel_start()
        logger.info(f"[ParquetFIFOReader] [ParallelStartAll] Started {len(readers_list)} readers with independent thread pools")

    @staticmethod 
    def parallel_join_all(readers_list):
        """
        Static method to join all parallel readers and ensure their end_time is set.
        """
        for reader in readers_list:
            if reader.producer_futures:
                # Wait for all futures to complete
                for future in reader.producer_futures:
                    future.result()  # This will block until the future completes
                if reader.end_time is None:
                    reader.end_time = time.perf_counter_ns()
                reader._finished = True
                logger.info(f"[ParquetFIFOReader] [ParallelJoin] [Table {reader.table_name}] Finished, end_time set")

    def get_fifo(self) -> FIFO:
        return self.fifo
    
    def join(self):
        """
        Join threads for this specific reader.
        In sequential mode, joins only this reader's threads and then starts the next reader in the chain.
        In parallel mode, joins all threads across all readers. 
        - THIS SHOULD BE CALLED BY MAIN THREAD ONLY!
        - Only FIFO forward() calls this function. MAIN THREAD calls FIFO forward().                                                                      
        """
        if not self._started.is_set():
            raise RuntimeError("ParquetFIFOReader not started yet")
        if self._finished:
            return
        # self._finished = True set the in join_all_threads() or join_one_reader()
        
        sequential_mode = os.environ.get("READER_SEQUENTIAL", "False") == "True"
        if sequential_mode:
            # In sequential mode, join the current reader
            join_one_reader(self)
            # Start the next reader in the chain
            if self.next_reader is not None:
                logger.info(f"[ParquetFIFOReader] [Table {self.table_name}] Starting next reader in chain: {self.next_reader.table_name}")
                self.next_reader.start_time = time.perf_counter_ns()
                self.next_reader.start()
            else:
                logger.info(f"[ParquetFIFOReader] [Table {self.table_name}] This was the last reader in the chain")
        else:
            # In parallel mode, join all threads across all readers
            join_all_threads()
    
    def is_finished(self) -> bool:
        return self._finished
    
    def get_num_rowgroups(self) -> int:
        return len(self.rg_sources)
    
    def get_rank_num_rowgroups(self) -> int:
        return self.rank_num_rowgroups
    
    def get_rank_range(self) -> tuple:
        assert self.rank_rowgroups
        return (min(self.rank_rowgroups), max(self.rank_rowgroups))

if __name__ == "__main__":
    logger_setup() # INFO level by default
    if len(sys.argv) < 5:
        print("Usage: python Parquet_FIFO.py <parquet_file> <num_threads> <table_name> <columns(comma-separated)>")
        print("Environment variables: WORLD_RANK, WORLD_SIZE (default: 0, 1)")
        sys.exit(1)
    
    parquet_file = sys.argv[1]
    num_threads = int(sys.argv[2])
    table_name = sys.argv[3]
    columns_arg = sys.argv[4]
    column_names = [c.strip() for c in columns_arg.split(',') if c.strip()]
    if not column_names:
        print("Error: at least one column required")
        sys.exit(1)
    
    # Get distributed parameters from environment or defaults
    world_rank = int(os.environ.get('WORLD_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    logger.info(f"[ParquetFIFOReader] Starting distributed Parquet FIFO reader: rank {world_rank}/{world_size}")
    
    mr = rmm.mr.CudaAsyncMemoryResource()
    rmm.mr.set_current_device_resource(mr)

    # Constructor with distributed parameters
    reader = ParquetFIFOReader(parquet_file, table_name, column_names, num_threads, world_rank, world_size)
    
    logger.info(f"Total rowgroups: {reader.get_num_rowgroups()}")
    logger.info(f"Rank {world_rank} rowgroups: {reader.get_rank_num_rowgroups()} (range: {reader.get_rank_range()})")
    logger.info(f"column_names: {reader.column_names}")
    logger.info(f"get_dtypes: {reader.get_dtypes()} -> pytorch dtype for each column with the same order as column_names")

    # Get the FIFO for consuming, but no do dequeue it before starting
    fifo = reader.get_fifo()
    
    # Start the producer threads, the finish() is called in each thread inside
    reader.start()
    
    # Start a consumer thread
    def tensor_consumer(fifo_queue, consumer_id=0):
        """Example consumer that drains the FIFO and reports received batches."""
        processed_count = 0
        while True:
            tensor_morsel = fifo_queue.forward()  # GET
            if tensor_morsel is None:  # Finished
                logger.info(f"[ConsumerLog] Consumer {consumer_id} terminating, total={processed_count}")
                break
            
            column_names = tensor_morsel.columns()
            for i, column_name in enumerate(column_names):
                tensor = tensor_morsel[i]
                if __debug__:
                    logger.debug(f"[ConsumerLog] Consumer {consumer_id}: tensor[{i}] for column[{i}]={column_name}  shape={tensor.shape} dtype={tensor.dtype} device={tensor.device}")
                if reader.is_decimal[column_name]:
                    col_schema = reader.column_schema[column_name]
                    if __debug__:
                        logger.debug(f"[ConsumerLog] Consumer {consumer_id}: tensor[{i}] for column[{i}]={column_name} logical_type={col_schema.logical_type}, DECIMAL's precision={col_schema.precision} scale={col_schema.scale}")
            processed_count += 1
            shapes = {column_name: list(tensor_morsel.get_by_colname(column_name).shape) for column_name in column_names}
            if __debug__:
                logger.debug(f"[ConsumerLog] Consumer {consumer_id} got batch #{processed_count} column_names={column_names} shapes={shapes}")
        return processed_count
    tensor_consumer(fifo,0)
    
    # Wait for all producers to finish
    reader.join()
    
    # Shutdown the thread pool
    nvtx.push_range(message = f"THREAD POOL SHUTDOWN {reader.table_name}")
    reader.thread_pool.shutdown(wait=False) 
    nvtx.pop_range()

    logger.info(f"[Done] FIFO drained. Remaining size={fifo.qsize()}")