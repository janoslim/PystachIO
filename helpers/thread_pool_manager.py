import threading
import logging
from concurrent.futures import ThreadPoolExecutor
import os
import ctypes
logger = logging.getLogger(__name__)

### Global Thread Pool Management ###
GLOBAL_THREAD_POOL = None
THREAD_POOL_LOCK = threading.Lock()

# Global collection of all thread futures across reader chains
GLOBAL_THREAD_FUTURES = []
GLOBAL_FUTURES_LOCK = threading.Lock()

def _set_reader_thread_name():
    """Set the Linux OS thread name (PR_SET_NAME = 15) so pidstat -t attributes reader pool CPU time."""
    ctypes.CDLL("libc.so.6").prctl(15, b"py_reader")

def get_global_thread_pool(max_workers):
    """Get or create the global thread pool with specified max workers."""
    global GLOBAL_THREAD_POOL
    num_threads = max(max_workers, int(os.environ.get("STORAGE_THREADS", 1)))
    with THREAD_POOL_LOCK:
        if GLOBAL_THREAD_POOL is None:
            GLOBAL_THREAD_POOL = ThreadPoolExecutor(
                max_workers=num_threads,
                thread_name_prefix="th", # use this as a index to streams
                initializer=_set_reader_thread_name
            )
            logger.info(f"[ThreadPoolManager] Created global thread pool with {num_threads} workers")
        else:
            logger.info(f"[ThreadPoolManager] Reusing existing global thread pool")
        return GLOBAL_THREAD_POOL

def shutdown_global_thread_pool():
    """Shutdown the global thread pool."""
    global GLOBAL_THREAD_POOL
    with THREAD_POOL_LOCK:
        if GLOBAL_THREAD_POOL is not None:
            GLOBAL_THREAD_POOL.shutdown(wait=True)
            GLOBAL_THREAD_POOL = None
            logger.info("[ThreadPoolManager] Global thread pool shut down")

def add_thread_future(future, reader, task_id):
    """Add a thread future to the global collection for batched joining."""
    global GLOBAL_THREAD_FUTURES
    with GLOBAL_FUTURES_LOCK:
        GLOBAL_THREAD_FUTURES.append((future, reader, task_id))

def join_one_reader(reader):
    """Join futures for a specific reader only, without destroying the thread pool."""
    global GLOBAL_THREAD_FUTURES
    with GLOBAL_FUTURES_LOCK:
        if not GLOBAL_THREAD_FUTURES:
            logger.warning(f"[ThreadPoolManager][join_one_reader][READER_SEQUENTIAL=True] [Table {reader.table_name}] No futures found in global list")
            return
        # Safety check: verify all futures belong to this reader
        for future, r, task_id in GLOBAL_THREAD_FUTURES:
          if r is not reader:
                logger.warning(f"[ThreadPoolManager][join_one_reader][READER_SEQUENTIAL=True] [Table {reader.table_name}] Found future from different reader object: {r.table_name}, expected all futures to be from reader object {id(reader)}")
        for future, r, task_id in GLOBAL_THREAD_FUTURES:
            future.result()  # Wait for the job to finish
            if __debug__:
                logger.debug(f"[ThreadPoolManager][join_one_reader][READER_SEQUENTIAL=True] [Table {reader.table_name}] Task {task_id} finished")
        # Update filtered bytes for this reader
        with reader._filtered_bytes_lock:
            if not reader.is_finished():
                for col_name, filtered_bytes in reader._filtered_bytes.items():
                    assert col_name in reader.storage_bytes
                    reader.storage_bytes[col_name][2] = filtered_bytes
                    logger.info(f"[ThreadPoolManager][join_one_reader][READER_SEQUENTIAL=True] [Table {reader.table_name}] Aggregated filtered bytes for column {col_name}: {filtered_bytes / (1 << 30):.2f} GB")
        reader._finished = True
        logger.info(f"[ThreadPoolManager][join_one_reader][READER_SEQUENTIAL=True] [Table {reader.table_name}] Reader completed")
        
        # Clear all futures since they all belonged to this reader
        futures_count = len(GLOBAL_THREAD_FUTURES)
        GLOBAL_THREAD_FUTURES.clear()
        logger.info(f"[ThreadPoolManager][join_one_reader][READER_SEQUENTIAL=True] [Table {reader.table_name}] Cleared {futures_count} futures from global list")

def join_all_threads():
    """Join all threads across all readers in the chain."""
    global GLOBAL_THREAD_FUTURES
    with GLOBAL_FUTURES_LOCK:
        logger.info(f"[ThreadPoolManager] Joining {len(GLOBAL_THREAD_FUTURES)} jobs (not threads...) across all readers")
        for future, reader, task_id in GLOBAL_THREAD_FUTURES:
            future.result()  # Wait for the job to finish
            if __debug__:
                logger.debug(f"[ThreadPoolManager][join_all_threads][READER_SEQUENTIAL=False] [Table {reader.table_name}] Task {task_id} finished")
        
        for _, reader, _ in GLOBAL_THREAD_FUTURES:
            if not reader.is_finished():
                with reader._filtered_bytes_lock:
                    for col_name, filtered_bytes in reader._filtered_bytes.items():
                        assert col_name in reader.storage_bytes
                        reader.storage_bytes[col_name][2] = filtered_bytes
                        logger.info(f"[ThreadPoolManager] [Table {reader.table_name}] Aggregated filtered bytes for column {col_name}: {filtered_bytes / (1 << 30):.2f} GB")
            reader._finished = True
            if __debug__:
                logger.debug(f"[ThreadPoolManager] [Table {reader.table_name}] All tasks finished, filtered bytes aggregated")
        GLOBAL_THREAD_FUTURES.clear() # Clear the futures list
        logger.info("[ThreadPoolManager][join_all_threads][READER_SEQUENTIAL=False] All reader tasks completed")

def clear_thread_futures():
    """Clear the global thread futures list (for cleanup)."""
    global GLOBAL_THREAD_FUTURES
    with GLOBAL_FUTURES_LOCK:
        GLOBAL_THREAD_FUTURES.clear()