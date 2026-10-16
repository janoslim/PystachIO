"""FlowCSD observation patch: records that do not change query execution.

* reset_pool_peak() at query start and print_query_window() at query end: the query window in
  CLOCK_MONOTONIC ns (time.perf_counter_ns is clock_gettime(CLOCK_MONOTONIC) on Linux) and the RMM
  cudaMallocAsync pool that serves cuDF and torch (run_tpch.py): its size and the peak bytes in use
  during the query. Reading and resetting a pool's statistics does not change its allocations.
* first_read_begin()/first_read_end() around every row-group read_parquet call (Parquet_FIFO): the start
  and end of the process's first call, printed by print_query_window(). libcudf creates its pinned host
  pool (LIBCUDF_PINNED_POOL_SIZE, 16 GiB by config.py) and other first-use state inside that call.
* dump_result(): the TPC-H query functions print only the first rows of their results; this saves
  all result tensors to the .npz file named by PYSTACHIO_RESULT_DUMP (nothing if unset).

The engine's own timers (ReaderWrapper.run) start after reset_pool_peak() and stop before the rest.
"""
import os
import threading
import time

import numpy as np
import rmm
from cuda.bindings import driver

_pool = None  # the current RMM pool, found once; False if allocations do not come from a pool
_first_read_ns = [-1, -1]  # start and end (perf_counter_ns) of the process's first read_parquet call
_first_read_lock = threading.Lock()


def current_pool():
    """The cudaMemPool behind RMM's current resource (CudaAsyncMemoryResource), or False."""
    global _pool
    if _pool is None:
        # RMM's Python API does not expose the pool handle; ask CUDA which pool a 1-byte buffer came from.
        probe = rmm.DeviceBuffer(size=1)
        error, pool = driver.cuPointerGetAttribute(driver.CUpointer_attribute.CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE,
                                                   probe.ptr)
        _pool = pool if error == driver.CUresult.CUDA_SUCCESS else False
    return _pool


def reset_pool_peak() -> None:
    """Zero the pool's used-memory high-water mark so it covers this query only.

    RMM primes the pool by allocating half of the free GPU memory at startup, which would
    otherwise dominate the mark.
    """
    pool = current_pool()
    if pool:
        driver.cuMemPoolSetAttribute(pool, driver.CUmemPool_attribute.CU_MEMPOOL_ATTR_USED_MEM_HIGH,
                                     driver.cuuint64_t(0))


def pool_bytes() -> tuple[int, int]:
    """(peak bytes in use since reset_pool_peak, bytes the pool holds now), or (-1, -1) without a pool."""
    pool = current_pool()
    if not pool:
        return -1, -1
    _, used_high = driver.cuMemPoolGetAttribute(pool, driver.CUmemPool_attribute.CU_MEMPOOL_ATTR_USED_MEM_HIGH)
    _, reserved = driver.cuMemPoolGetAttribute(pool, driver.CUmemPool_attribute.CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT)
    return int(used_high), int(reserved)


def first_read_begin() -> bool:
    """Called before every row-group read_parquet; True (and the start recorded) only for the process's first."""
    if _first_read_ns[0] >= 0:
        return False
    with _first_read_lock:
        if _first_read_ns[0] >= 0:
            return False
        _first_read_ns[0] = time.perf_counter_ns()
        return True


def first_read_end() -> None:
    """Called after the process's first read_parquet call returns."""
    _first_read_ns[1] = time.perf_counter_ns()


def print_query_window(query_start_ns: int, run_start_ns: int, query_end_ns: int) -> None:
    """Print the query window, the first read_parquet call and the pool numbers as one parseable line."""
    used_high, reserved = pool_bytes()
    print(f"# FLOWCSD query_start_ns={query_start_ns} run_start_ns={run_start_ns} "
          f"query_end_ns={query_end_ns} first_read_start_ns={_first_read_ns[0]} "
          f"first_read_end_ns={_first_read_ns[1]} pool_used_high_bytes={used_high} "
          f"pool_reserved_bytes={reserved}", flush=True)


def dump_result(tensors: dict) -> None:
    """Save {name: tensor} as numpy arrays in $PYSTACHIO_RESULT_DUMP (an .npz file)."""
    path = os.environ.get("PYSTACHIO_RESULT_DUMP")
    if not path:
        return
    np.savez(path, **{name: tensor.detach().cpu().numpy() for name, tensor in tensors.items()})
