# https://github.com/nboeschen/pytorchio/issues/158 i


# Figure 16: TPC-H Query 3 runtime executed on 2 GPUs with
# I/O performance: storage and network I/O bus bandwidth
# monitored over the query time with different optimizations

# (1,1)
READER_SEQUENTIAL=True SHUFFLE_TYPE=BLOCKING USE_BMON=True NUM_REPEATS=10 KVIKIO_NTHREADS=16 STORAGE_THREADS=8 TPCH_SCALE_FACTOR=400 TPCH_QUERY=3 python multiproc.py run_tpch.py 2

# (1,2)
READER_SEQUENTIAL=True STORAGE_BLOCKING=True USE_BMON=True NUM_REPEATS=10 KVIKIO_NTHREADS=16 STORAGE_THREADS=8 TPCH_SCALE_FACTOR=400 TPCH_QUERY=3 python multiproc.py run_tpch.py 2

# (1,3)
SHUFFLE_BUFFER_RIGHT=2 STORAGE_BLOCKING=True USE_BMON=True NUM_REPEATS=10 KVIKIO_NTHREADS=16 STORAGE_THREADS=8 TPCH_SCALE_FACTOR=400 TPCH_QUERY=3 python multiproc.py run_tpch.py 2

# (2,3)
SHUFFLE_BUFFER_RIGHT=2 USE_BMON=True NUM_REPEATS=10 KVIKIO_NTHREADS=16 STORAGE_THREADS=8 TPCH_SCALE_FACTOR=400 TPCH_QUERY=3 python multiproc.py run_tpch.py 2

# (2,2)
# Overlapping, but seq reader
READER_SEQUENTIAL=True  USE_BMON=True NUM_REPEATS=10 KVIKIO_NTHREADS=16 STORAGE_THREADS=8 TPCH_SCALE_FACTOR=400 TPCH_QUERY=3 python multiproc.py run_tpch.py 2

# (2,1)
# Overlapping, but single thread in storage and network

READER_SEQUENTIAL=True USE_BMON=True NUM_REPEATS=10 KVIKIO_NTHREADS=16 STORAGE_THREADS=1 TPCH_SCALE_FACTOR=400 TPCH_QUERY=3 python multiproc.py run_tpch.py 2