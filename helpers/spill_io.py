"""GPU <-> SSD spill partitions of the spill mode (SHUFFLE_TYPE=GRACE, patch 0003), through cuFile (GDS).

SpillPartitions keeps the P partitions of one operator input (a join side or partial groups). Partitions
0..k-1 are resident: their rows stay in GPU memory (hybrid hash join and aggregation). The other partitions go to
files on the data SSDs: append() collects whole morsels; once flush_bytes are pending, flush() groups them by
key % P and hands every spilled partition's rows to the I/O threads as chunks. An operator that chooses k after
its input is partitioned (GraceHashJoin) starts with a larger k and lowers it with lower_resident(). The join also
keeps the keys of the build rows it spills (spilled_keys) and gives its probe side the key filter built from them
(key_filter): the probe side's grouping then drops the rows of spilled partitions that cannot join, and the few rows
it keeps per spilled partition are held until a batch's worth is held (hold_rows), so requests stay large.
ChunkPrefetcher reads spilled partitions back in the order the operator consumes them, a window ahead.

I/O path (SpillIO): every drive has its own IO_THREADS_PER_DRIVE threads, each with its own cuFile-registered
cudaMalloc staging buffer of REQUEST_BYTES and its own CUDA stream. A write copies a chunk's columns into the
staging buffer (each column padded to 4 KiB) and calls cuFileWrite; a read calls cuFileRead and copies the columns
out. Data moves GPU <-> SSD by GDS P2PDMA, never through host memory. Up to IO_THREADS_PER_DRIVE requests are in
flight per drive while the main thread keeps partitioning (WRITE_BATCHES_IN_FLIGHT batches) or joining (reads
PREFETCH_BYTES ahead).

File layout: partition p's chunks cycle over the given directories starting at p % len(spill_dirs) (one directory:
one file per partition; directories on N SSDs: partitions and chunks are striped over all N). A chunk's columns are stored back to back, each
padded to 4 KiB, so every file offset and I/O length is a multiple of 4 KiB (O_DIRECT). Integer columns are
narrowed losslessly: the rows one batch writes to a partition store value - min in the smallest of uint8, int16,
int32 that holds their range (NARROW_DTYPES; floats unchanged), and a read adds min back. The host keeps the
index: per chunk its file, offset, row count, padded length and per column the stored dtype and offset.
"""
import ctypes
import math
import os
import queue
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import torch
from cuda.bindings import cufile, runtime

from helpers.flowcsd_observe import pool_unused_bytes, spill_stats

ALIGNMENT_BYTES = 4096  # O_DIRECT and cuFile: file offsets and I/O lengths in 4 KiB units
# One cuFile request = one chunk at most. 16 and 32 MiB requests read at the drives' 2.7-3.3 GiB/s with 2 or more
# in flight, 64 MiB ones slower (engines/pystachio/spill_io_sweep.py, 20261017T091957Z-pystachio-spill-io-sweep).
REQUEST_BYTES = 32 << 20
# Requests in flight per drive. The drive write profile (spill/03_drive_write_profile.sh) wrote 3.1-3.3 GiB/s per
# drive with 4 threads per drive; 4 threads shared by two drives wrote about 2.1 GB/s per drive (SF1000 q03 striped).
IO_THREADS_PER_DRIVE = 4
FLUSH_MIN_BYTES = 512 << 20     # pending morsel bytes before a flush, at least one request per partition
WRITE_BATCHES_IN_FLIGHT = 2     # double buffering: the GPU partitions batch n+1 while batch n is written
PREFETCH_BYTES = 1 << 30        # spilled chunks read ahead of the operator during the join or finalize
MAX_PARTITIONS = 255            # partition ids are uint8 in flush() (one 8-bit radix pass)
# Lossless narrowing of spilled integer columns: a chunk column stores value - offset (offset = its smallest value)
# in the first of these dtypes that holds its range and is narrower than the column; floats keep their dtype.
NARROW_DTYPES = [(torch.uint8, 255), (torch.int16, 32767), (torch.int32, 2**31 - 1)]
LIBCUFILE_SONAME = "libcufile.so.0"

_cufile = None  # the process's libcufile through ctypes, set up once by cufile_library()
_spill_io = None  # the process's I/O threads, created once by spill_io()


def aligned(nbytes: int) -> int:
    """nbytes rounded up to a multiple of ALIGNMENT_BYTES."""
    return -(-nbytes // ALIGNMENT_BYTES) * ALIGNMENT_BYTES


def spill_partition_count() -> int:
    """P of every spilling operator of the query: SPILL_PARTITIONS, which run_query.sh takes from spill_partitions.csv."""
    partitions = int(os.environ["SPILL_PARTITIONS"])
    if partitions < 1:
        raise ValueError(f"SPILL_PARTITIONS={partitions}: spill_partitions.csv has no row for this query and scale factor")
    return partitions


def resident_partition_count(partitions: int) -> int:
    """k: partitions 0..k-1 stay in GPU memory (SPILL_RESIDENT_PARTITIONS, from spill_partitions.csv). The hybrid
    join takes it as an upper bound and keeps fewer when they do not fit (GraceHashJoin.build_resident_joins)."""
    resident = int(os.environ["SPILL_RESIDENT_PARTITIONS"])
    if not 0 <= resident <= partitions:
        raise ValueError(f"SPILL_RESIDENT_PARTITIONS={resident} is not between 0 and SPILL_PARTITIONS={partitions}")
    return resident


def mapped_libcufile_paths() -> list[str]:
    """The libcufile files mapped into this process (from /proc/self/maps)."""
    paths = set()
    with open("/proc/self/maps") as maps:
        for line in maps:
            if "libcufile.so" in line:
                paths.add(line.split()[-1])
    return sorted(paths)


def cufile_library() -> ctypes.CDLL:
    """Open the cuFile driver once and return libcufile with cuFileWrite/cuFileRead typed for ctypes.

    The process must map one libcufile only: importing torch maps the pip wheel's copy, and kvikio (the Parquet
    reads) and cuda.bindings reuse it; a second copy would run a second cuFile driver instance. The two I/O
    calls go through ctypes on that copy because cuda.bindings 12.9.4's cufile.write/cufile.read drop cuFile's
    return value (the bytes moved, or a negative error code other than -1, which the binding does not raise).
    """
    global _cufile
    if _cufile is not None:
        return _cufile
    cufile.driver_open()  # reference counted: kvikio opens the same driver for the Parquet reads
    paths = mapped_libcufile_paths()
    if len(paths) != 1:
        raise RuntimeError(f"expected one libcufile in this process, found {paths}")
    library = ctypes.CDLL(LIBCUFILE_SONAME, mode=os.RTLD_NOLOAD | os.RTLD_LAZY)  # the copy already mapped
    for name in ("cuFileWrite", "cuFileRead"):
        function = getattr(library, name)
        # ssize_t cuFileWrite/cuFileRead(CUfileHandle_t, void *buffer, size_t size, off_t file_offset, off_t buffer_offset)
        function.restype = ctypes.c_ssize_t
        function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int64, ctypes.c_int64]
    _cufile = library
    return library


def drive_name(path: str) -> str:
    """Block device that holds path (e.g. nvme2n1), the per-drive key of the spill statistics."""
    device = os.stat(path).st_dev
    return os.path.basename(os.path.realpath(f"/sys/dev/block/{os.major(device)}:{os.minor(device)}"))


def wait_for(futures: list, direction: str) -> None:
    """Wait on the main thread for spill I/O requests; the time spent here is the exposed spill time."""
    start_ns = time.perf_counter_ns()
    for future in futures:
        future.result()
    spill_stats.add(f"spill_{direction}_exposed_ns", time.perf_counter_ns() - start_ns)


def gpu_bytes_available() -> int:
    """GPU memory this process can still allocate: free device memory plus what the RMM pool holds but does not use."""
    free_bytes, _ = torch.cuda.mem_get_info()
    return free_bytes + pool_unused_bytes()


def stored_layout(dtype: torch.dtype, value_range: tuple[int, int] | None) -> tuple[torch.dtype, int]:
    """(dtype the column is stored in, offset subtracted before storing) for values in value_range (None: float)."""
    if value_range is None:
        return dtype, 0
    low, high = value_range
    for narrow_dtype, largest in NARROW_DTYPES:
        if narrow_dtype.itemsize < dtype.itemsize and high - low <= largest:
            return narrow_dtype, low
    return dtype, 0


def value_ranges(column_sets: list[list[torch.Tensor]]) -> list[list[tuple[int, int] | None]]:
    """(min, max) of every integer column of each non-empty column set (None for float columns), one host sync."""
    bounds = []
    for columns in column_sets:
        for column in columns:
            if not column.dtype.is_floating_point:
                bounds.append(torch.stack(torch.aminmax(column)).to(torch.int64))
    values = torch.stack(bounds).tolist() if bounds else []
    ranges = []
    position = 0
    for columns in column_sets:
        column_ranges = []
        for column in columns:
            if column.dtype.is_floating_point:
                column_ranges.append(None)
            else:
                column_ranges.append((values[position][0], values[position][1]))
                position += 1
        ranges.append(column_ranges)
    return ranges


class StagingBuffer:
    """A GPU buffer registered with cuFile once, through which one I/O thread moves all its chunks.

    cuFile moves data between the SSD and a registered buffer directly; an unregistered buffer goes through
    cuFile's own bounce buffers, which made GPU -> SN770 writes about 3x slower in the GDS gate run
    (experimental_results/20261017T073632Z-spill-gds-gate). The buffer is cudaMalloc'd outside the RMM pool:
    cuFile registered an RMM pool (cudaMallocAsync) buffer but could not find it again to deregister it.
    """

    def __init__(self, size_bytes: int):
        error, pointer = runtime.cudaMalloc(size_bytes)
        if error != runtime.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaMalloc of the {size_bytes} B spill staging buffer failed: {error}")
        self.pointer = int(pointer)
        # A uint8 tensor over the memory: torch.as_tensor wraps it without copying (CUDA array interface).
        self.__cuda_array_interface__ = {"shape": (size_bytes,), "typestr": "|u1", "data": (self.pointer, False),
                                         "version": 3}
        self.tensor = torch.as_tensor(self, device="cuda")
        cufile.buf_register(self.pointer, size_bytes, 0)

    def close(self) -> None:
        cufile.buf_deregister(self.pointer)
        self.tensor = None
        runtime.cudaFree(self.pointer)


class SpillFile:
    """One spill file: O_DIRECT, registered with cuFile; chunks are appended at 4 KiB-aligned offsets."""

    def __init__(self, path: str):
        self.path = path
        self.fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_DIRECT, 0o644)
        self.drive = drive_name(path)
        self.descr = cufile.Descr()
        self.descr.type = cufile.FileHandleType.OPAQUE_FD
        self.descr.handle.fd = self.fd
        self.descr.fs_ops = 0
        self.handle = cufile.handle_register(self.descr.ptr)
        self.end_offset = 0  # where the next chunk goes

    def close(self) -> None:
        cufile.handle_deregister(self.handle)
        os.close(self.fd)


class SpillIO:
    """The process's spill I/O threads: every drive has its own IO_THREADS_PER_DRIVE threads (its own request
    queue), and every thread owns a registered staging buffer and a CUDA stream."""

    def __init__(self):
        self.cufile = cufile_library()
        self.staging_buffers = queue.SimpleQueue()  # each I/O thread takes one when it starts
        self.local = threading.local()
        self.drive_pools = {}  # drive -> ThreadPoolExecutor of its I/O threads

    def add_drive(self, drive: str) -> None:
        """Give one drive its I/O threads (once); called from operator constructors, before the query runs."""
        if drive in self.drive_pools:
            return
        # Allocated here, on the main thread before the query starts: the RMM pool keeps the GPU memory it grows
        # to, so a cudaMalloc later in the query could fail.
        for _ in range(IO_THREADS_PER_DRIVE):
            self.staging_buffers.put(StagingBuffer(REQUEST_BYTES))
        self.drive_pools[drive] = ThreadPoolExecutor(max_workers=IO_THREADS_PER_DRIVE,
                                                     thread_name_prefix=f"spill_io_{drive}",
                                                     initializer=self.start_thread)

    def start_thread(self) -> None:
        ctypes.CDLL("libc.so.6").prctl(15, b"py_spill")  # PR_SET_NAME = 15 for pidstat -t CPU attribution
        self.local.staging = self.staging_buffers.get()
        self.local.stream = torch.cuda.Stream()

    def write(self, spill_file: SpillFile, columns: list[torch.Tensor], ready: torch.cuda.Event, chunk: "Chunk"):
        """Queue one chunk write on its drive; the columns are complete once `ready` (main stream) fires."""
        return self.drive_pools[spill_file.drive].submit(self.write_now, spill_file, columns, ready, chunk)

    # The query runs under torch.inference_mode(), which is per thread: the staging buffers and destinations are
    # inference tensors, so the I/O threads' copies into them must run in inference mode too.
    @torch.inference_mode()
    def write_now(self, spill_file: SpillFile, columns: list[torch.Tensor], ready: torch.cuda.Event,
                  chunk: "Chunk") -> None:
        staging, stream = self.local.staging, self.local.stream
        stream.wait_event(ready)
        padded_bytes = 0
        with torch.cuda.stream(stream):
            for column, stored_dtype, offset in zip(columns, chunk.stored_dtypes, chunk.offsets):
                nbytes = column.shape[0] * stored_dtype.itemsize
                target = staging.tensor[padded_bytes:padded_bytes + nbytes]
                if stored_dtype == column.dtype:
                    target.copy_(column.view(torch.uint8))
                else:
                    target.view(stored_dtype).copy_(column - offset)  # 0 <= value - offset fits stored_dtype
                padded_bytes += aligned(nbytes)
        stream.synchronize()  # cuFile reads the staging buffer by DMA, outside any CUDA stream
        spill_stats.request_started("write", spill_file.drive)
        with torch.cuda.nvtx.range("spill_write"):
            written = self.cufile.cuFileWrite(spill_file.handle, staging.pointer, padded_bytes, chunk.file_offset, 0)
        spill_stats.request_finished("write", spill_file.drive, padded_bytes)
        if written != padded_bytes:
            raise OSError(f"cuFileWrite wrote {written} of {padded_bytes} bytes to {spill_file.path}")

    def read(self, spill_file: SpillFile, chunk: "Chunk", destinations: list[torch.Tensor],
             allocated: torch.cuda.Event):
        """Queue one chunk read on its drive into destinations (allocated on the main stream before `allocated`)."""
        return self.drive_pools[spill_file.drive].submit(self.read_now, spill_file, chunk, destinations, allocated)

    @torch.inference_mode()
    def read_now(self, spill_file: SpillFile, chunk: "Chunk", destinations: list[torch.Tensor],
                 allocated: torch.cuda.Event) -> None:
        staging, stream = self.local.staging, self.local.stream
        spill_stats.request_started("read", spill_file.drive)
        with torch.cuda.nvtx.range("spill_read"):
            read = self.cufile.cuFileRead(spill_file.handle, staging.pointer, chunk.padded_bytes, chunk.file_offset, 0)
        spill_stats.request_finished("read", spill_file.drive, chunk.padded_bytes)
        if read != chunk.padded_bytes:
            raise OSError(f"cuFileRead read {read} of {chunk.padded_bytes} bytes from {spill_file.path}")
        # The destinations' memory may have been used by earlier kernels of the main stream: copy after them.
        stream.wait_event(allocated)
        padded_bytes = 0
        with torch.cuda.stream(stream):
            for destination, stored_dtype, offset in zip(destinations, chunk.stored_dtypes, chunk.offsets):
                nbytes = chunk.rows * stored_dtype.itemsize
                stored = staging.tensor[padded_bytes:padded_bytes + nbytes]
                if stored_dtype == destination.dtype:
                    destination.view(torch.uint8).copy_(stored)
                else:
                    destination.copy_(stored.view(stored_dtype))  # widen, then undo the offset
                    destination.add_(offset)
                padded_bytes += aligned(nbytes)
        stream.synchronize()  # destinations complete; this thread's staging buffer is free for its next request


def spill_io() -> SpillIO:
    """The process's I/O threads, created on first use (an operator constructor, before the query runs)."""
    global _spill_io
    if _spill_io is None:
        _spill_io = SpillIO()
    return _spill_io


@dataclass
class Chunk:
    """Where one chunk of a spilled partition is, how many rows it holds and how its columns are stored."""
    file_index: int  # which of the partition's files (one per spill directory)
    file_offset: int
    rows: int
    padded_bytes: int
    stored_dtypes: list[torch.dtype]  # per column: its dtype, or a narrower one holding value - offset
    offsets: list[int]


class SpillPartitions:
    """The P partitions of one operator input: k resident in GPU memory, the rest in spill files."""

    def __init__(self, directories: str, name: str, partitions: int, resident: int, dtypes: list[torch.dtype],
                 key_index: int):
        """directories: one spill directory, or several separated by ':' to stripe chunks over them (files are
        <name>-p<partition>.bin). dtypes: one per column; key_index: the column whose value % partitions picks
        a row's partition. Partitions below `resident` stay in GPU memory (lower_resident() may spill some later).
        """
        if partitions > MAX_PARTITIONS:
            raise ValueError(f"{partitions} partitions: at most {MAX_PARTITIONS} (partition ids are uint8)")
        directory_list = directories.split(":")
        for directory in directory_list:
            # run_query.sh creates the per-run directories on the data SSDs. Never fall back to another place:
            # a relative or missing path would put the spill on the OS disk.
            if not os.path.isabs(directory) or not os.path.isdir(directory):
                raise FileNotFoundError(f"spill directory {directory!r} does not exist (run_query.sh creates it)")
        self.io = spill_io()
        self.name = name
        self.partitions = partitions
        self.resident = resident
        self.dtypes = dtypes
        self.key_index = key_index
        self.row_bytes = sum(dtype.itemsize for dtype in dtypes)
        # About one request per partition per flush.
        self.flush_bytes = max(FLUSH_MIN_BYTES, partitions * REQUEST_BYTES)
        # Every partition gets its files, also the resident ones (lower_resident() may spill them): opened and
        # registered here, before the query runs. run_query.sh deletes them, empty ones too.
        self.files = {}
        for partition in range(partitions):
            self.files[partition] = [SpillFile(os.path.join(directory, f"{name}-p{partition:03d}.bin"))
                                     for directory in directory_list]
        for spill_file in self.files[0]:
            self.io.add_drive(spill_file.drive)  # one file per directory: every drive of this input
        self.chunks = {partition: [] for partition in range(partitions)}  # host index, in file order
        self.resident_morsels = [[] for _ in range(resident)]
        # An operator that uses resident rows at once (the hybrid join's probe side) sets this to
        # fn(partition, columns); otherwise resident rows are kept (copied out of the batch) in resident_morsels.
        self.use_resident_rows = None
        # The hybrid join's build side sets this to a list: the key column of every row it spills (flush,
        # lower_resident) is kept there until the join has built its key filter from them.
        self.spilled_keys = None
        # The hybrid join's probe side gets the KeyFilter over the spilled partitions' build keys (none if k = P):
        # partition_pending then drops the rows of spilled partitions whose key it rejects.
        self.key_filter = None
        self.pending = []  # morsels (lists of column tensors) not partitioned yet
        self.pending_bytes = 0
        self.write_batches = deque()  # the chunk-write futures of every flushed batch still in flight
        # With a key filter a batch keeps few rows of each spilled partition: hold_rows() collects them here
        # (partition -> compact copies) until a batch's worth is held, so that writes stay about REQUEST_BYTES each.
        self.held = {partition: [] for partition in range(partitions)}
        self.held_bytes = 0
        # Present (0) for every spill operator, also one that spills nothing; record_narrowing adds to them.
        spill_stats.add("spill_unnarrowed_bytes", 0)
        spill_stats.add("spill_narrowed_bytes", 0)

    def append(self, columns: list[torch.Tensor]) -> None:
        """Keep one morsel; partition the pending morsels once flush_bytes are pending."""
        self.pending.append(columns)
        self.pending_bytes += columns[0].shape[0] * self.row_bytes
        if self.pending_bytes >= self.flush_bytes:
            self.flush()

    def flush(self) -> None:
        """Group the pending rows by key % P: resident partitions' rows stay on the GPU, the others are written
        (with a key filter: held, see hold_rows).

        The main thread's time here goes to spill_partition_ns (grouping, which applies the key filter if there is one,
        compact copies of resident and held rows, cutting chunks and queueing their writes), except the hybrid join's
        use of resident rows (it probes them: join work) and the wait for older write batches (spill_write_exposed_ns).
        """
        if not self.pending:
            return
        start_ns = time.perf_counter_ns()
        with torch.cuda.nvtx.range("spill_partition"):
            columns, rows_per_partition = self.partition_pending()
            ready = torch.cuda.Event()
            ready.record()
            spilled = []      # (partition, columns) to write
            rows_to_use = []  # (partition, columns) of resident partitions, for use_resident_rows
            first_row = 0
            for partition, rows in enumerate(rows_per_partition):
                rows_of_partition = [column[first_row:first_row + rows] for column in columns]
                first_row += rows
                if rows == 0:
                    continue
                if partition >= self.resident:
                    spilled.append((partition, rows_of_partition))
                    if self.spilled_keys is not None:
                        # A compact copy: a slice would keep the whole batch in GPU memory until the filter is built.
                        self.spilled_keys.append(rows_of_partition[self.key_index].clone())
                elif self.use_resident_rows is None:
                    # A compact copy, so that the batch is freed once its spilled rows are written.
                    self.resident_morsels[partition].append([column.clone() for column in rows_of_partition])
                else:
                    rows_to_use.append((partition, rows_of_partition))
            if self.key_filter is None:
                self.write_batches.append(self.write_partitions(spilled, ready))
        spill_stats.add("spill_partition_ns", time.perf_counter_ns() - start_ns)
        if self.key_filter is not None:
            self.hold_rows(spilled)
        # The hybrid join probes its resident partitions' rows now, while this batch is written.
        for partition, rows_of_partition in rows_to_use:
            self.use_resident_rows(partition, rows_of_partition)
        # Double buffering: the GPU partitions the next batch while this one is written; wait for older batches.
        while len(self.write_batches) > WRITE_BATCHES_IN_FLIGHT:
            wait_for(self.write_batches.popleft(), "write")

    def hold_rows(self, spilled: list[tuple[int, list[torch.Tensor]]]) -> None:
        """Keep compact copies of a filtered batch's spilled rows. The filter leaves each spilled partition a small
        share of a batch, and writing it at once would make small requests (and small chunks to probe later). Before
        they would make the held rows exceed flush_bytes, the held rows are written (write_held), so at most one batch
        is held. The held rows are the second write buffer: with them, one write batch is in flight, not
        WRITE_BATCHES_IN_FLIGHT, and the probe phase needs no more GPU memory than without the filter.
        """
        new_bytes = sum(columns[0].shape[0] for _, columns in spilled) * self.row_bytes
        if self.held_bytes + new_bytes > self.flush_bytes:
            self.write_held()
        start_ns = time.perf_counter_ns()
        with torch.cuda.nvtx.range("spill_partition"):
            for partition, columns in spilled:
                self.held[partition].append([column.clone() for column in columns])
        spill_stats.add("spill_partition_ns", time.perf_counter_ns() - start_ns)
        self.held_bytes += new_bytes

    def write_held(self) -> None:
        """Queue the held rows' writes as one write batch, each partition's copies concatenated (requests of about
        REQUEST_BYTES). First waits until the previous write batch is on the SSD (the held rows are its double buffer).
        """
        if self.held_bytes == 0:
            return
        while self.write_batches:
            wait_for(self.write_batches.popleft(), "write")
        start_ns = time.perf_counter_ns()
        with torch.cuda.nvtx.range("spill_partition"):
            spilled = []
            for partition, pieces in self.held.items():
                if not pieces:
                    continue
                columns = [torch.cat([piece[index] for piece in pieces]) for index in range(len(self.dtypes))]
                spilled.append((partition, columns))
            self.held = {partition: [] for partition in range(self.partitions)}
            self.held_bytes = 0
            ready = torch.cuda.Event()
            ready.record()  # the concatenated copies were made on this stream
            self.write_batches.append(self.write_partitions(spilled, ready))
        spill_stats.add("spill_partition_ns", time.perf_counter_ns() - start_ns)

    def partition_pending(self) -> tuple[list[torch.Tensor], list[int]]:
        """The pending rows reordered so that every partition's rows are contiguous, and the rows per partition.

        The partition ids are uint8, so the stable sort is one 8-bit radix pass in CUB (a histogram, its prefix sum
        and a scatter) instead of the full 64-bit sort that int64 ids get. With a key filter, the rows of spilled
        partitions whose key it rejects get the id P: they sort last and are not gathered, so the batch keeps only
        rows that are probed at once or may join later.
        """
        columns = [torch.cat([morsel[index] for morsel in self.pending]) for index in range(len(self.dtypes))]
        # Pending morsels may come straight from a reader FIFO, allocated on a reader thread's stream: their
        # memory goes back to the pool on that stream when dropped, and a reader's next decode may reuse it at
        # once. The cat on this stream must have read them before that.
        torch.cuda.current_stream().synchronize()
        self.pending = []
        self.pending_bytes = 0
        partition_of_row, filter_events = self.partition_ids(columns[self.key_index])
        rows_per_id = torch.bincount(partition_of_row, minlength=self.partitions + 1).tolist()  # host sync
        order = torch.argsort(partition_of_row, stable=True)
        if filter_events is not None:
            self.record_filter(rows_per_id, filter_events)
        dropped_rows = rows_per_id[self.partitions]
        kept_order = order[:order.shape[0] - dropped_rows]
        return [column[kept_order] for column in columns], rows_per_id[:self.partitions]

    def partition_ids(self, keys: torch.Tensor) -> tuple[torch.Tensor, tuple | None]:
        """uint8 partition id of every row (key % P) and, with a key filter, the CUDA events around its kernel.

        With a key filter one kernel computes the ids and gives the rows of spilled partitions (k..P-1) whose key the
        filter rejects the id P. Its GPU time is read from the events once the stream has synchronized anyway.
        """
        if self.key_filter is None:
            return torch.remainder(keys, self.partitions).to(torch.uint8), None
        filter_start = torch.cuda.Event(enable_timing=True)
        filter_end = torch.cuda.Event(enable_timing=True)
        filter_start.record()
        partition_of_row = self.key_filter.partition_ids(keys, self.partitions, self.resident)
        filter_end.record()
        return partition_of_row, (filter_start, filter_end)

    def record_filter(self, rows_per_id: list[int], filter_events: tuple) -> None:
        """Key filter statistics of one batch: its kernel's GPU time, and the rows routed to spilled partitions before
        and after the filter (rows_per_id[P] = the rows it dropped)."""
        filter_start, filter_end = filter_events
        dropped_rows = rows_per_id[self.partitions]
        kept_spilled_rows = sum(rows_per_id[self.resident:self.partitions])
        spill_stats.add("spill_filter_apply_ns", round(filter_start.elapsed_time(filter_end) * 1e6))
        spill_stats.add("spill_filter_probe_rows_before", kept_spilled_rows + dropped_rows)
        spill_stats.add("spill_filter_probe_rows_after", kept_spilled_rows)

    def lower_resident(self, resident: int) -> None:
        """Keep only partitions 0..resident-1 in GPU memory: the kept rows of the partitions above are written like
        a flushed batch and freed once on the SSD. For an operator that chooses k after partitioning (GraceHashJoin).
        """
        start_ns = time.perf_counter_ns()
        ready = torch.cuda.Event()
        ready.record()  # the kept copies were made on this stream
        spilled = []
        for partition in range(resident, self.resident):
            for columns in self.take_resident(partition):
                spilled.append((partition, columns))
                if self.spilled_keys is not None:
                    self.spilled_keys.append(columns[self.key_index])  # already a compact copy
        writes = self.write_partitions(spilled, ready)
        if writes:
            self.write_batches.append(writes)
        self.resident = resident
        self.resident_morsels = self.resident_morsels[:resident]
        spill_stats.add("spill_partition_ns", time.perf_counter_ns() - start_ns)

    def write_partitions(self, spilled: list[tuple[int, list[torch.Tensor]]], ready: torch.cuda.Event) -> list:
        """Queue the writes of some partitions' rows; their value ranges, which pick the narrowed widths, come back to
        the host in one sync."""
        ranges = value_ranges([columns for _, columns in spilled])
        writes = []
        for (partition, columns), column_ranges in zip(spilled, ranges):
            writes += self.write_rows(partition, columns, ready, column_ranges)
        return writes

    def write_rows(self, partition: int, columns: list[torch.Tensor], ready: torch.cuda.Event,
                   column_ranges: list[tuple[int, int] | None]) -> list:
        """Queue one spilled partition's rows as chunks of equal size, alternating over its files. Integer columns
        are stored narrowed (column_ranges: their (min, max) over these rows; None for floats)."""
        layout = [stored_layout(dtype, value_range) for dtype, value_range in zip(self.dtypes, column_ranges)]
        stored_dtypes = [stored_dtype for stored_dtype, _ in layout]
        offsets = [offset for _, offset in layout]
        rows = columns[0].shape[0]
        # Rows per chunk such that the stored columns, each padded to 4 KiB, fit in one request.
        rows_per_request = (REQUEST_BYTES - len(self.dtypes) * ALIGNMENT_BYTES) // sum(
            dtype.itemsize for dtype in stored_dtypes)
        chunk_count = math.ceil(rows / rows_per_request)
        rows_per_chunk = math.ceil(rows / chunk_count)
        writes = []
        for chunk_start in range(0, rows, rows_per_chunk):
            chunk_columns = [column[chunk_start:chunk_start + rows_per_chunk] for column in columns]
            chunk_rows = chunk_columns[0].shape[0]
            padded_bytes = sum(aligned(chunk_rows * dtype.itemsize) for dtype in stored_dtypes)
            file_index = (partition + len(self.chunks[partition])) % len(self.files[partition])
            spill_file = self.files[partition][file_index]
            chunk = Chunk(file_index, spill_file.end_offset, chunk_rows, padded_bytes, stored_dtypes, offsets)
            self.chunks[partition].append(chunk)
            writes.append(self.io.write(spill_file, chunk_columns, ready, chunk))
            spill_file.end_offset += padded_bytes
        self.record_narrowing(rows, stored_dtypes)
        return writes

    def record_narrowing(self, rows: int, stored_dtypes: list[torch.dtype]) -> None:
        """Spilled bytes at the original and at the stored widths, and rows per (column position, stored width)."""
        spill_stats.add("spill_unnarrowed_bytes", rows * self.row_bytes)
        spill_stats.add("spill_narrowed_bytes", rows * sum(dtype.itemsize for dtype in stored_dtypes))
        for position, stored_dtype in enumerate(stored_dtypes):
            spill_stats.add(f"spill_width_{self.name}_col{position}_bytes{stored_dtype.itemsize}_rows", rows)

    def finish_writes(self) -> None:
        """Partition what is pending, write what is held and wait until every chunk is on the SSD."""
        self.flush()
        self.write_held()
        while self.write_batches:
            wait_for(self.write_batches.popleft(), "write")

    def partition_rows(self, partition: int) -> int:
        """Rows of one spilled partition."""
        return sum(chunk.rows for chunk in self.chunks[partition])

    def take_resident(self, partition: int) -> list[list[torch.Tensor]]:
        """The kept morsels of one resident partition (and forget them)."""
        morsels = self.resident_morsels[partition]
        self.resident_morsels[partition] = []
        return morsels

    def close(self) -> None:
        """Release the cuFile handles and file descriptors; run_query.sh deletes the files."""
        for partition_files in self.files.values():
            for spill_file in partition_files:
                spill_file.close()


class ChunkPrefetcher:
    """Reads spilled partitions back in the order an operator consumes them, PREFETCH_BYTES ahead of it."""

    def __init__(self, reads: list[tuple[SpillPartitions, int]]):
        """reads: (spill partitions, partition) in the order the operator takes them; take() returns every chunk
        of each in turn."""
        self.order = [(spill, partition, chunk) for spill, partition in reads for chunk in spill.chunks[partition]]
        self.next_index = 0
        self.scheduled = deque()  # (future, destinations, padded bytes) in consumption order
        self.scheduled_bytes = 0
        self.schedule()

    def schedule(self) -> None:
        """Queue reads until PREFETCH_BYTES are scheduled ahead."""
        while self.next_index < len(self.order) and (not self.scheduled or self.scheduled_bytes < PREFETCH_BYTES):
            spill, partition, chunk = self.order[self.next_index]
            self.next_index += 1
            destinations = [torch.empty(chunk.rows, dtype=dtype, device="cuda") for dtype in spill.dtypes]
            allocated = torch.cuda.Event()
            allocated.record()
            future = spill.io.read(spill.files[partition][chunk.file_index], chunk, destinations, allocated)
            self.scheduled.append((future, destinations, chunk.padded_bytes))
            self.scheduled_bytes += chunk.padded_bytes

    def take(self) -> list[torch.Tensor]:
        """The next chunk's columns, waiting for its read if it is not done yet."""
        future, destinations, padded_bytes = self.scheduled.popleft()
        self.scheduled_bytes -= padded_bytes
        self.schedule()
        wait_for([future], "read")
        return destinations
