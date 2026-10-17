#!/usr/bin/env python3
import os

"""
Centralized configuration 
Use from other modules:
import config
config.apply_defaults(os.environ)                   # to ensure defaults are present in current environment
cur_env = config.apply_defaults(os.environ.copy())  # to get a copy with defaults
"""

GPU_2_NIC_MAPPING = {
    "GPU-fbcffbe0-c1fa-e698-919e-feefc9323321": ["mlx5_0"], # 07:00
    "GPU-07b5e8e7-7cd3-3e22-9300-35bae8829939": ["mlx5_1"], # 0F:00

    "GPU-1ef889d3-1332-a22e-88bc-54910f4e754a": ["mlx5_2"], # 47:00
    "GPU-15e58378-e584-57e2-fe70-5bb20adcf902": ["mlx5_3"], # 4E:00

    "GPU-61738c82-fd4e-4799-9c10-fee20c7b63da": ["mlx5_4"], # 87:00
    "GPU-72f859cf-3444-d3e5-bbb3-8fd8204f7dc1": ["mlx5_5"], # 90:00

    "GPU-b0e0689b-08ff-bdd9-c14a-2cbcc0004d3b": ["mlx5_6"], # B7:00
    "GPU-5b0c7adb-6993-9760-39d6-f0ac1fb3cf97": ["mlx5_7"], # BD:00

    "GPU-7528a6ce-2165-8d82-a92b-c9a20f5dab7a": ["mlx5_1"],
    "GPU-2c6143d8-3fd8-5a24-4b5c-da81c100f0c1": ["mlx5_0"],
}

GPU_2_SSD_MAPPING = {
    "GPU-fbcffbe0-c1fa-e698-919e-feefc9323321": "/xssd/0/", # 07:00
    "GPU-07b5e8e7-7cd3-3e22-9300-35bae8829939": "/xssd/4/", # 0F:00

    "GPU-1ef889d3-1332-a22e-88bc-54910f4e754a": "/xssd/2/", # 47:00
    "GPU-15e58378-e584-57e2-fe70-5bb20adcf902": "/xssd/7/", # 4E:00

    "GPU-61738c82-fd4e-4799-9c10-fee20c7b63da": "/xssd/3/", # 87:00
    "GPU-72f859cf-3444-d3e5-bbb3-8fd8204f7dc1": "/xssd/5/", # 90:00

    "GPU-b0e0689b-08ff-bdd9-c14a-2cbcc0004d3b": "/xssd/6/", # B7:00
    "GPU-5b0c7adb-6993-9760-39d6-f0ac1fb3cf97": "/xssd/8/", # BD:00

    "GPU-7528a6ce-2165-8d82-a92b-c9a20f5dab7a": "/datasets/",
    "GPU-2c6143d8-3fd8-5a24-4b5c-da81c100f0c1": "/datasets/",
}

# Default values (strings to mirror os.environ usage)
def apply_defaults():
    cur_env = os.environ.copy()
    cur_env["PATH"] = f"~/miniconda3/envs/pytorchio-env/bin/:{cur_env['PATH']}"
    
    cur_env.setdefault("MASTER_ADDR", "10.0.2.51")
    cur_env.setdefault("MASTER_PORT", "1324")
    cur_env.setdefault("TCP_STORE_PORT", "1235")
    cur_env.setdefault("PYTHONOPTIMIZE", "1") # optimize without assert and __debug__
    
    # NCCL_COMM_ID=ip:port
    cur_env.setdefault("NCCL_RAS_ENABLE", "0")
    cur_env.setdefault("NCCL_IB_DISABLE", "0")
    cur_env.setdefault("NCCL_P2P_DISABLE", "1")
    cur_env.setdefault("NCCL_SHM_DISABLE", "1")
    cur_env.setdefault("NCCL_NVB_DISABLE", "1")
    cur_env.setdefault("NCCL_PXN_DISABLE", "1")
    cur_env.setdefault("NCCL_DEBUG", "")
    
    cur_env.setdefault("USE_OWN_STDOUT", "True")
    cur_env.setdefault("STDOUT_TEMPLATE", "out_{}.txt")
    cur_env.setdefault("USE_BMON", "False")
    cur_env.setdefault("BMON_TIME", "50")
    
    cur_env.setdefault("NUM_REPEATS","3")
    cur_env.setdefault("SHUFFLE_TYPE", "OVERLAPPED") # [BLOCKING,OVERLAPPED,OVERLAPPED_GROUPS,GRACE] GRACE = SSD spill mode (patch 0003)
    cur_env.setdefault("SHUFFLE_STREAMS", "2")
    cur_env.setdefault("SHUFFLE_SEND_TYPE", "SEND_PER_RANK") # [SEND_PER_RANK, COMB_SEND]
    cur_env.setdefault("SHUFFLE_RECV_TYPE", "RECV_PER_RANK") # [RECV_PER_RANK, COMB_RECV] # COMB_RECV is not benefitial for in-memory raw shuffle performance!
    cur_env.setdefault("SHUFFLE_PROBE_TYPE", "FULL") # [MATCHES_ONLY, FULL, NOOP]
    cur_env.setdefault("SHUFFLE_PROBE_SCHED", "INVERTED") # [STANDARD,INVERTED]
    cur_env.setdefault("SHUFFLE_RIGHT_REPEAT", "5") # Number of repeats of the right table morsels
    cur_env.setdefault("HASH_SIZE_NAIVE", "False") # [False,True]
    cur_env.setdefault("SHUFFLE_BUILD_STREAMING_HT", "0") # 0 means number of ht is dynamic based on collisions. >0 means a fixed number of ht is created and 
    cur_env.setdefault("USE_RMM", "True")
    cur_env.setdefault("USE_NET_OPT_FLAGS", "True")
    cur_env.setdefault("EXTRA_OUTPUT", "0")
    
    cur_env.setdefault("TPCH_SCALE_FACTOR","1")
    cur_env.setdefault("TPCH_QUERY", "queryX")
    cur_env.setdefault("SYNC_RESULT", "True") # This sync is not timed
    
    # Storage
    cur_env.setdefault("STORAGE_LOGGING_LEVEL", "WARNING") # WARNING, INFO, DEBUG https://docs.python.org/3/library/logging.html#logging-levels
    cur_env.setdefault("STORAGE_THREADS", "8")
    cur_env.setdefault("STORAGE_SPECIAL_PATH", "") # STORAGE_BASE_PATH = GPU_2_SSD_MAPPING + STORAGE_SPECIAL_PATH
    cur_env.setdefault("STORAGE_BLOCKING", "False") # if True, all data is materialized in-memory
    cur_env.setdefault("PRUNING_FLAG", "False") 
    cur_env.setdefault("READER_SEQUENTIAL", "False")   # [AblationStudy] With this, each reader owns the thread pool explicitly, not sharing. So the reading bandwidth is not high.
    cur_env.setdefault("MULTI_SSD_READER", "False") # reader experiment: 1 GPU, >1 SSD
    # Spill tree (patch 0003): a byte budget per reader replaces FIFO_QUEUE_MAX_PERCENTAGE; row groups are submitted
    # with credits so that at most this many decoded bytes per table are read ahead of the plan (Parquet_FIFO.py).
    cur_env.setdefault("FIFO_QUEUE_MAX_BYTES", str(256 * 1024 * 1024))

    # SSD spill (SHUFFLE_TYPE=GRACE, patch 0003; run_query.sh --mode grace-spill sets them)
    cur_env.setdefault("SPILL_PARTITIONS", "0") # partitions of every spilling operator (spill_partitions.csv); 0 = the query has none
    cur_env.setdefault("SPILL_RESIDENT_PARTITIONS", "0") # k: partitions 0..k-1 stay in GPU memory (hybrid; spill_partitions.csv); the join keeps at most k, as many as fit
    cur_env.setdefault("SPILL_PARTITION_BUDGET_BYTES", str(4 * 1024 * 1024 * 1024)) # GPU state of one partition (join build or group-by finalize)
    cur_env.setdefault("SPILL_DIR_LINEITEM", "") # per-run directory (or ':'-separated, striped) for lineitem-derived partitions
    cur_env.setdefault("SPILL_DIR_OTHER_TABLES", "") # same for partitions derived from the other tables
    
    # CUDF
    cur_env.setdefault("KVIKIO_COMPAT_MODE", "OFF")
    cur_env.setdefault("KVIKIO_NTHREADS", "16")
    cur_env.setdefault("KVIKIO_GDS_THRESHOLD", "512") # Basically all-in-GDS
    cur_env.setdefault("LIBCUDF_ALLOCATE_HOST_AS_PINNED_THRESHOLD", str(1024 * 1024 * 1024 * 1024)) # 1024GB (as infinity)
    cur_env.setdefault("LIBCUDF_KERNEL_PINNED_COPY_THRESHOLD", str(1024 * 1024 * 1024 * 1024)) # 1024GB (as infinity)
    cur_env.setdefault("LIBCUDF_PINNED_POOL_SIZE", str(16 * 1024 * 1024 * 1024)) # 16GB pinned
    # cur_env.setdefault("LIBCUDF_PINNED_POOL_SIZE", str(1 * 1024 * 1024 * 1024)) # 1GB pinned
    
    if cur_env["USE_NET_OPT_FLAGS"] == "True":
        # cur_env.setdefault("CUDA_MODULE_LOADING", "EAGER")
        cur_env["TORCH_NCCL_HIGH_PRIORITY"] = "True"
        cur_env["NCCL_MIN_NCHANNELS"] = "32"
        # cur_env["NCCL_MAX_NCHANNELS"] = "32"
        # cur_env["NCCL_MIN_CTAS"] = "32"
        # cur_env["NCCL_MAX_CTAS"] = "32"
        # this one understandably only makes sense for large transfer sizes
        cur_env["NCCL_P2P_NET_CHUNKSIZE"] = "524288"
        # cur_env["NCCL_P2P_NET_CHUNKSIZE"] = "1048576"
        cur_env["NCCL_LOCAL_REGISTER"] = "1"
        # cur_env["NCCL_BUFFSIZE"] = "16777216"
        # cur_env["NCCL_CGA_CLUSTER_SIZE"] = "8"
        # cur_env["NCCL_SOCKET_NTHREADS"] = "8"
        # cur_env["NCCL_NSOCKS_PERTHREAD"] = "8"
    
    if cur_env["SHUFFLE_TYPE"] == "BLOCKING":
        cur_env["STORAGE_THREADS"] = "1"

    return cur_env