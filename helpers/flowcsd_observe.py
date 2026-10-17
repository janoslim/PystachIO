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
* spill_stats (spill mode, patch 0003): the GPU<->SSD spill I/O of the query, printed on the same line: bytes,
  busy time and cuFile calls per direction and per drive, the exposed time (main thread waiting for it), the
  partition step's time, when the spill phases ended, how many partitions stayed resident, the join's probe-side
  key filter and the narrowing of spilled columns. The line is also printed when the query raises
  (query_end_ns=-1), so a failed query still reports what it spilled.

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


def pool_unused_bytes() -> int:
    """Bytes the RMM pool holds but does not use now (0 without a pool); the spill join counts them as free."""
    pool = current_pool()
    if not pool:
        return 0
    _, used = driver.cuMemPoolGetAttribute(pool, driver.CUmemPool_attribute.CU_MEMPOOL_ATTR_USED_MEM_CURRENT)
    _, reserved = driver.cuMemPoolGetAttribute(pool, driver.CUmemPool_attribute.CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT)
    return int(reserved) - int(used)


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


class SpillStats:
    """Spill totals (spill mode, patch 0003), added to by the spill I/O threads and the main thread.

    Per direction ("write", "read"): bytes as cuFile moved them (4 KiB-padded), cuFile calls, busy time (wall time
    with at least one request in flight; per drive and over all drives, so overlapping requests count once) and
    exposed time (the query's main thread waiting for spill I/O to finish). Drives are block device names; per drive
    also the cuFile calls (spill_<drive>_<direction>_requests: requests are not a fixed size, see spill_io.py).
    spill_io_busy_ns: wall time with any spill request in flight, write or read (the union of the two direction
    clocks: a read-ahead that overlaps writes counts once, so it can be less than their sum).
    Main thread only: spill_partition_ns (time grouping rows into partitions and queueing their writes, waits
    excluded), the spill operator's phase ends (spill_*_ns, perf_counter_ns like query_start_ns; one spill
    operator per query), spill_resident_partitions (the k it kept), the join's key filter over the build keys of
    its spilled partitions (spill_filter_*: kind and size, none when k = P (no size field, the rest 0), build time,
    apply time = GPU time of the partition step's id kernel that applies it (inside spill_partition_ns), probe rows
    routed to spilled partitions before and after it, measured false positives per million) and the narrowing of
    spilled columns (spill_unnarrowed_bytes, spill_narrowed_bytes, spill_width_<side>_col<i>_bytes<w>_rows).
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._in_flight = {}   # field prefix -> requests in flight
        self._busy_since = {}  # field prefix -> perf_counter_ns when its first request in flight started
        self.fields = {}       # FLOWCSD field name -> integer
        for direction in ("write", "read"):
            for name in ("bytes", "busy_ns", "exposed_ns"):
                self.fields[f"spill_{direction}_{name}"] = 0
            self.fields[f"spill_cufile_{direction}s"] = 0
        self.fields["spill_io_busy_ns"] = 0
        self.fields["spill_partition_ns"] = 0

    def add(self, name: str, value: int) -> None:
        with self._lock:
            self.fields[name] = self.fields.get(name, 0) + value

    def set(self, name: str, value: int) -> None:
        with self._lock:
            self.fields[name] = value

    def mark(self, name: str) -> None:
        """Record now as the end of a spill phase (e.g. spill_input_partitioned_ns)."""
        self.set(name, time.perf_counter_ns())

    @staticmethod
    def busy_clocks(direction: str, drive: str) -> tuple[str, ...]:
        """Field prefixes whose busy clock a request keeps running: its direction over all drives, its direction on
        its drive, and spill_io (any spill request)."""
        return (f"spill_{direction}", f"spill_{drive}_{direction}", "spill_io")

    def request_started(self, direction: str, drive: str) -> None:
        now_ns = time.perf_counter_ns()
        with self._lock:
            for clock in self.busy_clocks(direction, drive):
                if self._in_flight.get(clock, 0) == 0:
                    self._busy_since[clock] = now_ns
                self._in_flight[clock] = self._in_flight.get(clock, 0) + 1

    def request_finished(self, direction: str, drive: str, nbytes: int) -> None:
        now_ns = time.perf_counter_ns()
        with self._lock:
            for clock in self.busy_clocks(direction, drive):
                self._in_flight[clock] -= 1
                if self._in_flight[clock] == 0:
                    busy_name = f"{clock}_busy_ns"
                    self.fields[busy_name] = self.fields.get(busy_name, 0) + now_ns - self._busy_since[clock]
            # Bytes per direction and per drive only: the bytes of any spill I/O are write + read.
            for prefix in (f"spill_{direction}", f"spill_{drive}_{direction}"):
                bytes_name = f"{prefix}_bytes"
                self.fields[bytes_name] = self.fields.get(bytes_name, 0) + nbytes
            self.fields[f"spill_cufile_{direction}s"] += 1
            requests_name = f"spill_{drive}_{direction}_requests"
            self.fields[requests_name] = self.fields.get(requests_name, 0) + 1


spill_stats = SpillStats()


def print_query_window(query_start_ns: int, run_start_ns: int, query_end_ns: int) -> None:
    """Print the query window, the first read_parquet call, the pool numbers and the spill totals as one parseable
    line. query_end_ns = -1: the query raised (ReaderWrapper.run prints the line before re-raising)."""
    used_high, reserved = pool_bytes()
    spill_fields = " ".join(f"{name}={value}" for name, value in sorted(spill_stats.fields.items()))
    print(f"# FLOWCSD query_start_ns={query_start_ns} run_start_ns={run_start_ns} "
          f"query_end_ns={query_end_ns} first_read_start_ns={_first_read_ns[0]} "
          f"first_read_end_ns={_first_read_ns[1]} pool_used_high_bytes={used_high} "
          f"pool_reserved_bytes={reserved} {spill_fields}", flush=True)


def dump_result(tensors: dict) -> None:
    """Save {name: tensor} as numpy arrays in $PYSTACHIO_RESULT_DUMP (an .npz file)."""
    path = os.environ.get("PYSTACHIO_RESULT_DUMP")
    if not path:
        return
    np.savez(path, **{name: tensor.detach().cpu().numpy() for name, tensor in tensors.items()})
