import sys
import os
import datetime
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
import torch
import config
from modules.query_wrapper import ReaderWrapper
from modules.filter_module import FilterCondition, FilterOp, TableFilter

def run_standalone_reader(iteration, store, target_device, scale_factor):
    """
    Standalone reader
    """
    # Ensure environment has default config values
    storage_threads = int(os.environ.get("STORAGE_THREADS", "1"))
    STORAGE_BASE_PATH = os.environ.get("STORAGE_BASE_PATH", "/xssd/0")
    print(f"# [StandaloneReader] Using STORAGE_BASE_PATH: {STORAGE_BASE_PATH}")
    MULTI_SSD_PATHS = [STORAGE_BASE_PATH]
    num_ssds = 1
    if os.environ.get("MULTI_SSD_READER", "False") != "False":
        num_ssds = int(os.environ.get("MULTI_SSD_READER"))
        assert num_ssds >= 1
        # ordered unique paths from config, keep only /xssd
        all_paths = list(dict.fromkeys(config.GPU_2_SSD_MAPPING.values())) #  ['/xssd/5/', '/xssd/4/', '/xssd/2/', '/xssd/6/']
        xssd_paths = [p for p in all_paths if "/xssd" in p]
        assert STORAGE_BASE_PATH in xssd_paths, f"STORAGE_BASE_PATH {STORAGE_BASE_PATH} not in {xssd_paths}"
        xssd_paths.remove(STORAGE_BASE_PATH)
        # if '/xssd/4/' in xssd_paths:
        #     xssd_paths.remove('/xssd/4/')  # UGLY fix to remove some slow SSD
        MULTI_SSD_PATHS += xssd_paths[:num_ssds-1]
        print(f"# [StandaloneReader] Using MULTI_SSD_READER with {num_ssds} paths: {MULTI_SSD_PATHS}")

    assert len(MULTI_SSD_PATHS) == num_ssds
    table_infos = []
    for idx, ssd_path in enumerate(MULTI_SSD_PATHS):
        base_folder = f"{ssd_path}/tpch-sf{scale_factor}/"
        table_infos.append(
            (f"lineitem_{idx}", f"{base_folder}/lineitem.parquet", storage_threads,
            ["l_partkey", "l_suppkey"],
            # "l_orderkey","l_extendedprice","l_discount","l_shipdate"],
            TableFilter({
                # 3: [FilterCondition(FilterOp.GT, datetime.date(1995, 3, 15))]  # l_shipdate
            }))
        )

    reader_wrapper = ReaderWrapper(store, target_device, table_infos=table_infos)
    def tmp_function():
        return { "local_result": 0, "time_ns": float('inf') }  # float('inf') to avoid division by zero
    result_str = reader_wrapper.run(tmp_function)
    # TODO: (jigao) maybe I will add selectivity, #RG readed, and others later on 
    if iteration == 0:
        print(f"query,iteration,algo,{reader_wrapper.header_str}")
    print(f"reader,{iteration},reader,{result_str}")