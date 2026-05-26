import sys
import os
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
import pandas as pd
import time
import humanize

import torch
import torch.distributed as dist

from helpers.Parquet_FIFO import ParquetFIFOReader, CUDF_PARQUET_METADATA_CACHE
from helpers.thread_pool_manager import join_all_threads, clear_thread_futures
from helpers.beemon import Beemon
from modules.single_node_join import SingleNodeJoin
from modules.single_thread_naive import ShuffleWithJoinNaive
from modules.single_thread_shuffle_join_multistream import ShuffleWithJoin_MultiStream
from helpers.CuDFReader import CuDFReader

import logging
logger = logging.getLogger(__name__)

class ReaderWrapper:
    header_str = "sf,nodes,node_id,storage_threads,local_result,total_result,local_bytes_actual,local_bytes_uncompressed,local_bytes_shuffled,global_bytes_actual,global_bytes_uncompressed,all_files_uncomp_bytes,time_meta_ns,time_storage_ns,time_afterstorage_ns,time_part_ns,time_shuffle_ns,time_buildprobe_ns,time_finalize_agg_ns,time_ns,NCCL_IB_DISABLE,NCCL_P2P_DISABLE,STORAGE_BLOCKING,STORAGE_BASE_PATH,READER_SEQUENTIAL,SHUFFLE_STREAMS,SHUFFLE_RECV_TYPE,SHUFFLE_PROBE_TYPE,SHUFFLE_PROBE_SCHED,HASH_SIZE_NAIVE,USE_RMM,USE_NET_OPT_FLAGS,EXTRA_OUTPUT"
    """
    A convenience wrapper for query plans on parquet data.
    table_infos should be a list of "name, filepath, threads, cols, filters" tuples.
    This Wrapper is more flexible, since the interaction of joins can be defined inside a given function,
    while still having the convenience of collected metrics etc..
    read_full_infos = None --> Only the last two tables in table_infos are read partially, the rest are assumed to be replicated and read in full
    read_full_infos = list of table_name --> Tables with these names in table_infos will be read in full, other partitioned 
    """
    def __init__(self, store, target_device, *, table_infos,read_full_infos=None):
        self.WORLD_SIZE = int(os.environ['WORLD_SIZE'])
        self.WORLD_RANK = int(os.environ['RANK'])
        self.SF = int(os.environ["TPCH_SCALE_FACTOR"])
        
        self.store = store
        self.target_device = target_device

        clear_thread_futures()
        # Create the ParquetFIFOReader, time-consuming preparation: metadata caching...
        self.beemon = Beemon()

        self.total_storage_threads = 0
        use_cudf_native_reader = os.environ.get("USE_CUDF_NATIVE_READER", "False").lower() == "true"

        self.readers = []
        if use_cudf_native_reader:
            for idx, (name, filepath, storage_threads, cols, table_filter) in enumerate(table_infos):
                reader = CuDFReader(filepath, name, cols)
                self.total_storage_threads += 1  # CuDFReader uses 1 thread equivalent
                print(f"## Reader {idx} ({reader.table_name}): CuDFReader")
                self.readers.append(reader)
        else:
            for idx, (name, filepath, storage_threads, cols, table_filter) in enumerate(table_infos):
                if os.environ["SHUFFLE_TYPE"] == "BLOCKING":
                    storage_threads = 1
                self.total_storage_threads = storage_threads
                if read_full_infos is not None:
                    full_read_mode = (name in read_full_infos)    # all tables listed are full read mode
                else:
                    full_read_mode = (idx < len(table_infos) - 2) # only last two reader are not full read mode
                self.readers.append(ParquetFIFOReader(filepath, name, cols, num_threads=storage_threads,
                                               world_rank=self.WORLD_RANK, world_size=self.WORLD_SIZE,
                                               table_filter=table_filter, beemon=self.beemon, full_read_mode=full_read_mode, prune_flag=os.environ.get("PRUNING_FLAG")=="True"))

        # chain all readers (only for ParquetFIFOReader)
        if not use_cudf_native_reader:
            for readeridx,reader in enumerate(self.readers[:-1]):
                reader.set_next_reader(self.readers[readeridx+1])
        for idx, reader in enumerate(self.readers):
            logger.info(f"  Reader {idx} ({reader.table_name}): {reader.num_threads} threads (reused)")

    def run(self, callable):
        # synchronize nodes
        dist.barrier()

        use_cudf_native_reader = os.environ.get("USE_CUDF_NATIVE_READER", "False").lower() == "true"

        # Start ALL readers immediately - no waiting between them
        # The thread pool will handle scheduling naturally
        start = time.perf_counter_ns()
        
        if use_cudf_native_reader:
            logger.info(f"[ReaderWrapper] Using CuDFReader for {len(self.readers)} readers")
            for reader in self.readers:
                reader.start()
        else:
            reader_sequential_mode = os.environ.get("READER_SEQUENTIAL", "False") == "True"
            if reader_sequential_mode:
                if os.environ["SHUFFLE_TYPE"] == "BLOCKING" or os.environ["STORAGE_BLOCKING"] == "True":
                    logger.info(f"[ReaderWrapper] ReaderSequential enabled - starting and joining each reader sequentially")
                    for reader in self.readers:
                        reader.start_time = time.perf_counter_ns()
                        reader.start()
                        reader.join()
                        logger.info(f"[ReaderWrapper] Joined reader: {reader.table_name}")
                        # if reader != self.readers[-1]:
                        #     time.sleep(0.3) # just adding time-gaps for bmon plotting, this is not for benchmarking
                else:
                    logger.info(f"[ReaderWrapper] ReaderSequential enabled - starting first reader, others will be chained")
                    # Start only the first reader, others will be started by the previous reader's join()
                    first_reader = self.readers[0]
                    first_reader.start_time = time.perf_counter_ns()
                    first_reader.start()
                    logger.info(f"[ReaderWrapper] Started first reader: {first_reader.table_name}")
            else:
                if len(self.readers) > 1 and os.environ.get("MULTI_SSD_READER", "False") != "False":
                    # parallel_start for independent execution: for multi-SSDs and single GPU
                    logger.info(f"[ReaderWrapper] Using parallel_start() for {len(self.readers)} readers with independent thread pools")
                    ParquetFIFOReader.parallel_start_all(self.readers)
                else:
                    # Start all readers at once - they submit to the shared thread pool
                    #   "Start" is just submitted, but not get scheduled for I/O yet!!!
                    for reader in self.readers:
                        reader.start()  # Each reader submits its threads immediately
                    logger.info(f"[ReaderWrapper] Started all {len(self.readers)} readers immediately, threads queued in shared pool")
                if os.environ["SHUFFLE_TYPE"] == "BLOCKING" or os.environ["STORAGE_BLOCKING"] == "True":
                    join_all_threads()

        # how to join, when to call what, etc. configured in callable (e.g. see query3)
        results = callable()

        if not use_cudf_native_reader and "OVERLAPPED" in os.environ["SHUFFLE_TYPE"] and os.environ["STORAGE_BLOCKING"] != "True":
            if len(self.readers) > 1 and os.environ.get("MULTI_SSD_READER", "False") != "False":
                ParquetFIFOReader.parallel_join_all(self.readers)
            else:
                join_all_threads() # or join all readers
        end = time.perf_counter_ns()

        if use_cudf_native_reader:
            # Simplified metrics for CuDFReader
            time_meta_ns = 0  # No metadata caching for CuDFReader, TODO:(jigao) fix the right timer if needed
        else:
            time_meta_ns = sum([CUDF_PARQUET_METADATA_CACHE[reader.parquet_file][1] for reader in self.readers])
            
        assert all(reader.prepare_time is not None for reader in self.readers), "reader prepare_time=None"
        time_ns = end-start # all operations
        time_prepare_ns = sum([reader.prepare_time for reader in self.readers]) # Partitioning, Pruning, Syncing, logging.
        time_storage_ns = (self.readers[-1].end_time - start) + time_prepare_ns
        time_afterstorage_ns = results["time_ns"] # query processing time after storage (or during if overlap) = part+shuffle+buildprobe+finalize_agg
        time_part_ns = results.get("time_part_ns",-1)
        time_shuffle_ns =  results.get("time_shuffle_ns",-1)
        time_buildprobe_ns = results.get("time_buildprobe_ns",-1)
        time_finalize_agg_ns = results.get("time_finalize_agg_ns", -1)

        max_times = torch.tensor([time_meta_ns, time_storage_ns, time_afterstorage_ns, time_part_ns, time_shuffle_ns, time_buildprobe_ns, time_finalize_agg_ns, time_ns],device="cpu")
        join_tuples_total = torch.tensor([results["local_result"]],device="cpu")
        dist.all_reduce(max_times, op=dist.ReduceOp.MAX)
        dist.all_reduce(join_tuples_total, op=dist.ReduceOp.SUM)

        bus_factor = (self.WORLD_SIZE-1)/self.WORLD_SIZE
        local_storage_bytes = torch.tensor([sum(actual_bytes for actual_bytes, _, _, _, _ in reader.storage_bytes.values()) for reader in self.readers], device="cpu")
        local_uncomp_bytes = torch.tensor([sum(uncomp_bytes for _, uncomp_bytes, _, _, _ in reader.storage_bytes.values()) for reader in self.readers], device="cpu")
        if len(self.readers) >= 2:
            # for complex plans, only the query function can know the shuffled bytes (and should annotate results)
            left_shuffled, right_shuffled = results['shuffled_bytes']
            local_shuffled_bytes = torch.tensor([ left_shuffled, right_shuffled ])*bus_factor
        else:
            # Handle the single-reader case
            local_shuffled_bytes = torch.tensor([
                sum(shuffled_bytes for _, _, shuffled_bytes, _, _ in self.readers[-1].storage_bytes.values())
            ])*bus_factor
        global_storage_bytes = local_storage_bytes.detach().clone()
        global_uncomp_bytes = local_uncomp_bytes.detach().clone()
        global_shuffled_bytes = local_shuffled_bytes.detach().clone()
        dist.all_reduce(global_storage_bytes, op=dist.ReduceOp.SUM)
        dist.all_reduce(global_uncomp_bytes, op=dist.ReduceOp.SUM)
        dist.all_reduce(global_shuffled_bytes, op=dist.ReduceOp.SUM)
        file_uncomp_bytes = [sum(col_file_uncompressed_bytes for _, _, _, _, col_file_uncompressed_bytes in reader.storage_bytes.values()) for reader in self.readers] # this is file level, no need for dist.all_reduce
        all_files_uncomp_bytes = sum(file_uncomp_bytes)

        if os.environ.get("MULTI_SSD_READER", "False") == "False": # if MULTI_SSD_READER, then skipped
            if len(self.readers) >= 2:
                print(f"# Storage bytes: local={humanize.naturalsize(local_storage_bytes[-2], binary=True)} (left), {humanize.naturalsize(local_storage_bytes[-1], binary=True)} (right) | "
                                       f"total={humanize.naturalsize(global_storage_bytes[-2], binary=True)} (left), {humanize.naturalsize(global_storage_bytes[-1], binary=True)} (right)")
                print(f"# Uncompressed bytes: local={humanize.naturalsize(local_uncomp_bytes[-2], binary=True)} (left), {humanize.naturalsize(local_uncomp_bytes[-1], binary=True)} (right) | "
                                            f"total={humanize.naturalsize(global_uncomp_bytes[-2], binary=True)} (left), {humanize.naturalsize(global_uncomp_bytes[-1], binary=True)} (right)")
                print(f"# Shuffled bytes: local={humanize.naturalsize(local_shuffled_bytes[-2], binary=True)} (left), {humanize.naturalsize(local_shuffled_bytes[-1], binary=True)} (right) | "
                                        f"total={humanize.naturalsize(global_shuffled_bytes[-2], binary=True)} (left), {humanize.naturalsize(global_shuffled_bytes[-1], binary=True)} (right)")
            else:
                # Handle the single-reader case
                print(f"# Storage bytes: local={humanize.naturalsize(local_storage_bytes[0], binary=True)}"
                                       f"total={humanize.naturalsize(global_storage_bytes[0], binary=True)}")
                print(f"# Uncompressed bytes: local={humanize.naturalsize(local_uncomp_bytes[0], binary=True)}"
                                            f"total={humanize.naturalsize(global_uncomp_bytes[0], binary=True)}")
        print(f"# Storage bytes: local={humanize.naturalsize(local_storage_bytes.sum().item(), binary=True)} | "
                               f"total={humanize.naturalsize(global_storage_bytes.sum().item(), binary=True)}")
        print(f"# Uncompressed bytes: local={humanize.naturalsize(local_uncomp_bytes.sum().item(), binary=True)} | "
                                     f"total={humanize.naturalsize(global_uncomp_bytes.sum().item(), binary=True)}")
        global_time_storage_ns = max_times[1]
        print(f"# Pruning+Partitioning / Storage Time in %: {(time_prepare_ns/time_storage_ns*100):.2f}%")
        print(f"# [Storage][Local]  actual BW               : {local_storage_bytes.sum().item()/(1<<30) / (time_storage_ns/1e9):.2f} GiB/s")
        print(f"# [Storage][Global] actual BW               : {global_storage_bytes.sum().item()/(1<<30) / (global_time_storage_ns/1e9):.2f} GiB/s")
        print(f"# [Storage][Local]  effective BW            : {local_uncomp_bytes.sum().item()/(1<<30) / (time_storage_ns/1e9):.2f} GiB/s")
        print(f"# [Storage][Global] effective BW            : {global_uncomp_bytes.sum().item()/(1<<30) / (global_time_storage_ns/1e9):.2f} GiB/s")
        print(f"# [Storage][Global] effective BW + Pruning  : {all_files_uncomp_bytes/(1<<30) / (global_time_storage_ns/1e9):.2f} GiB/s")
        
        if global_storage_bytes.sum().item() > 0:
            compression_ratio = global_uncomp_bytes.sum().item() / global_storage_bytes.sum().item()
            print(f"# [Storage][Global] Compression ratio       : {compression_ratio:.2f}")
        else:
            print(f"# [Storage][Global] Compression ratio       : N/A (no storage bytes)")
            
        if global_uncomp_bytes.sum().item() > 0:
            pruning_ratio = all_files_uncomp_bytes / global_uncomp_bytes.sum().item()
            print(f"# [Storage][Global] Pruning ratio           : {pruning_ratio:.2f}")
        else:
            print(f"# [Storage][Global] Pruning ratio           : N/A (no uncompressed bytes)")

        global_time_afterstorage_ns = max_times[2]
        # print(f"# [Shuffle][Local]  BW                      : {local_shuffled_bytes.sum().item()/(1<<30) / (time_afterstorage_ns/1e9):.2f} GiB/s")
        print(f"# [Shuffle][Local]  BusBW                   : {local_shuffled_bytes.sum().item()/(1<<30) / (time_afterstorage_ns/1e9):.2f} GiB/s")
        # print(f"# [Shuffle][Global] BW                      : {global_shuffled_bytes.sum().item()/(1<<30) / (global_time_afterstorage_ns/1e9):.2f} GiB/s")

        global_time_ns = max_times[7]
        print(f"# [Query][Local]    actual BW               : {(local_storage_bytes.sum().item()/(1<<30)) / (global_time_ns/1e9):.2f} GiB/s")
        print(f"# [Query][Local]    effective BW            : {(local_uncomp_bytes.sum().item()/(1<<30)) / (global_time_ns/1e9):.2f} GiB/s")
        print(f"# [Query][Local]    QueryBW                 : {(local_uncomp_bytes.sum().item()*bus_factor/(1<<30)) / (global_time_ns/1e9):.2f} GiB/s")
        print(f"# [Query][Global]   actual BW               : {(global_storage_bytes.sum().item()/(1<<30)) / (global_time_ns/1e9):.2f} GiB/s")
        print(f"# [Query][Global]   effective BW            : {(global_uncomp_bytes.sum().item()/(1<<30)) / (global_time_ns/1e9):.2f} GiB/s")
        print(f"# [Query][Global]   effective BW + Pruning  : {(all_files_uncomp_bytes/(1<<30)) / (global_time_ns/1e9):.2f} GiB/s")

        env_var_info = ",".join(os.environ[key] for key in ["NCCL_IB_DISABLE","NCCL_P2P_DISABLE","STORAGE_BLOCKING","STORAGE_BASE_PATH","READER_SEQUENTIAL","SHUFFLE_STREAMS","SHUFFLE_RECV_TYPE",
                                            "SHUFFLE_PROBE_TYPE","SHUFFLE_PROBE_SCHED","HASH_SIZE_NAIVE","USE_RMM","USE_NET_OPT_FLAGS", "EXTRA_OUTPUT"])

        return f"{self.SF},{self.WORLD_SIZE},{self.WORLD_RANK},{self.total_storage_threads},{results['local_result']},{join_tuples_total[0]},{local_storage_bytes.sum().item()},{local_uncomp_bytes.sum().item()},{local_shuffled_bytes.sum().item()},{global_storage_bytes.sum().item()},{global_uncomp_bytes.sum().item()},{all_files_uncomp_bytes},{max_times[0]},{global_time_storage_ns},{global_time_afterstorage_ns},{max_times[3]},{max_times[4]},{max_times[5]},{max_times[6]},{global_time_ns},{env_var_info}"
