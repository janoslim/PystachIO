#!/usr/bin/env python3
import faulthandler
import math
import os
import time
from datetime import timedelta
import sys
import pandas as pd
import traceback
import cupy as cp
from helpers.beemon import Beemon
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroupNCCL

import gc
import rmm
from rmm.allocators.torch import rmm_torch_allocator

from helpers.util import cub_unique_consecutive
from helpers.nccl_init import nccl_init, nccl_init_sync

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <target_device_id>")
        exit(1)

    target_device = int(sys.argv[1])
    GPU_MEMORY = cp.cuda.Device().mem_info[1]  # [free, total] in bytes
    if os.environ.get("USE_RMM", "False") == "True":
        if os.environ.get("USE_RMM_MANAGED", "False") == "True":
            mr = rmm.mr.ManagedMemoryResource()
        else:
            mr = rmm.mr.CudaAsyncMemoryResource()
        rmm.mr.set_current_device_resource(mr)
        torch.cuda.memory.change_current_allocator(rmm_torch_allocator)

    # tmp fix for rmm issue
    import tpch.query1
    import tpch.query2
    import tpch.query3
    import tpch.query4
    import tpch.query5
    import tpch.query6
    import tpch.query7
    import tpch.query8
    import tpch.query9
    import tpch.query10
    import tpch.query11
    import tpch.query12
    import tpch.query13
    import tpch.query14
    import tpch.query15
    import tpch.query16
    import tpch.query17
    import tpch.query18
    import tpch.query19
    import tpch.query20
    import tpch.line_order_shuffle
    import tpch.var_line_order_shuffle
    import tpch.line_part_shuffle
    import tpch.reader

    LOCAL_RANK = int(os.environ['LOCAL_RANK'])
    WORLD_SIZE = int(os.environ['WORLD_SIZE'])
    WORLD_RANK = int(os.environ['RANK'])
    MASTER_IP = os.environ['MASTER_ADDR']
    TCP_STORE_PORT = int(os.environ['TCP_STORE_PORT'])
    repeats = int(os.environ["NUM_REPEATS"])
    scale_factor = int(os.environ["TPCH_SCALE_FACTOR"])
    tpch_query = os.environ["TPCH_QUERY"]

    store = dist.TCPStore(MASTER_IP, TCP_STORE_PORT, WORLD_SIZE, WORLD_RANK == 0, timedelta(seconds=30))
    dist.init_process_group("cuda:nccl,cpu:gloo", store=store, rank=WORLD_RANK, world_size=WORLD_SIZE, device_id=torch.device(target_device))

    faulthandler.enable()
    if WORLD_SIZE > 1:
        nccl_init(target_device)
        nccl_init_sync(target_device)
        torch.cuda.synchronize(target_device)

    # print(f"Using GPU: {os.environ.get("CUDA_VISIBLE_DEVICES")}, {target_device}")
    beemon = Beemon()
    try:
        with torch.jit.optimized_execution(False):
            with torch.inference_mode():
                for i in range(repeats):
                    beemon.start(world_rank=WORLD_RANK, num_repeat=i) # skip the previous: CURRENT_REPEAT
                    try:
                        torch.cuda.nvtx.range_push(f"REPEAT={i} {tpch_query} SF={scale_factor}")
                        if tpch_query == "query1" or tpch_query == "1":
                            tpch.query1.run_query1_on_parquet(i, store, target_device, scale_factor)
                        elif tpch_query == "query2" or tpch_query == "2":
                            tpch.query2.run_query2_on_parquet(i, store, target_device, scale_factor)
                        elif tpch_query == "query3" or tpch_query == "3":
                            tpch.query3.run_query3_on_parquet(i, store, target_device, scale_factor)
                        elif tpch_query == "query4" or tpch_query == "4":
                            tpch.query4.run_query4_on_parquet(i, store, target_device, scale_factor)
                        elif tpch_query == "query5" or tpch_query == "5":
                            tpch.query5.run_query5_on_parquet(i, store, target_device, scale_factor)
                        elif tpch_query == "query6" or tpch_query == "6":
                            tpch.query6.run_query6_on_parquet(i, store, target_device, scale_factor)
                        elif tpch_query == "query7" or tpch_query == "7":
                            tpch.query7.run_query7_on_parquet(i, store, target_device, scale_factor)
                        elif tpch_query == "query8" or tpch_query == "8":
                            tpch.query8.run_query8_on_parquet(i, store, target_device, scale_factor)
                        elif tpch_query == "query9" or tpch_query == "9":
                            tpch.query9.run_query9_on_parquet(i, store, target_device, scale_factor)
                        elif tpch_query == "query10" or tpch_query == "10":
                            tpch.query10.run_query10_on_parquet(i, store, target_device, scale_factor)
                        elif tpch_query == "query11" or tpch_query == "11":
                            tpch.query11.run_query11_on_parquet(i, store, target_device, scale_factor)
                        elif tpch_query == "query12" or tpch_query == "12":
                            tpch.query12.run_query12_on_parquet(i, store, target_device, scale_factor)
                        elif tpch_query == "query13" or tpch_query == "13":
                            tpch.query13.run_query13_on_parquet(i, store, target_device, scale_factor)
                        elif tpch_query == "query14" or tpch_query == "14":
                            tpch.query14.run_query14_on_parquet(i, store, target_device, scale_factor)
                        elif tpch_query == "query15" or tpch_query == "15":
                            tpch.query15.run_query15_on_parquet(i, store, target_device, scale_factor)
                        elif tpch_query == "query16" or tpch_query == "16":
                            tpch.query16.run_query16_on_parquet(i, store, target_device, scale_factor)
                        elif tpch_query == "query17" or tpch_query == "17":
                            tpch.query17.run_query17_on_parquet(i, store, target_device, scale_factor)
                        elif tpch_query == "query18" or tpch_query == "18":
                            tpch.query18.run_query18_on_parquet(i, store, target_device, scale_factor)
                        elif tpch_query == "query19" or tpch_query == "19":
                            tpch.query19.run_query19_on_parquet(i, store, target_device, scale_factor)
                        elif tpch_query == "query20" or tpch_query == "20":
                            tpch.query20.run_query20_on_parquet(i, store, target_device, scale_factor)
                        elif tpch_query == "line_order_shuffle":
                            tpch.line_order_shuffle.run_line_order_shuffle(i, store, target_device, scale_factor)
                        elif tpch_query == "var_line_order_shuffle" or tpch_query == "vlos":
                            tpch.var_line_order_shuffle.run_line_order_shuffle(i, store, target_device, scale_factor)
                        elif tpch_query == "line_part_shuffle":
                            tpch.line_part_shuffle.run_line_part_shuffle(i, store, target_device, scale_factor)
                        elif tpch_query == "reader": # standalone reader
                            tpch.reader.run_standalone_reader(i, store, target_device, scale_factor)
                        else:
                            print(f"Error: Unknown query: {tpch_query}")
                        torch.cuda.nvtx.range_pop()
                        torch.cuda.empty_cache()
                        gc.collect()
                    except Exception as e:
                        print(f"Exception in iteration {i}: {e}")
                        print("Stack trace:")
                        traceback.print_exc()
                        print("Continuing to next iteration...")
                        continue
                    time.sleep(0.2) # creating bmon 0.2s gap
                    beemon.stop()
    except Exception as e:
        print(f"Fatal exception: {e}")
        print("Stack trace:")
        traceback.print_exc()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        beemon.stop()
    os._exit(0)
