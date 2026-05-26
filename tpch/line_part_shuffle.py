import sys
import os
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))

import torch
import torch.distributed as dist

from helpers import util
from modules.query_wrapper import ReaderWrapper
from modules.filter_module import FilterCondition, FilterOp, TableFilter
from modules.single_thread_naive import ShuffleWithJoinNaive
from modules.single_thread_shuffle_join_multistream import ShuffleWithJoin_MultiStream

def run_line_part_shuffle(iteration, store, target_device, scale_factor):
    WORLD_SIZE = int(os.environ['WORLD_SIZE'])
    if os.environ["SHUFFLE_TYPE"] == "BLOCKING":
        SHUFFLE_MODULE_TYPE = ShuffleWithJoinNaive
    elif "OVERLAPPED" in os.environ["SHUFFLE_TYPE"]:
        SHUFFLE_MODULE_TYPE = ShuffleWithJoin_MultiStream
    else:
        print(f"Error, unknown SHUFFLE_TYPE={os.environ["SHUFFLE_TYPE"]}")
        return
    storage_threads = int(os.environ["STORAGE_THREADS"])

    base_folder = f"{os.environ['STORAGE_BASE_PATH']}/tpch-sf{scale_factor}"

    # Query: select count(*) from lineitem, part where l_partkey = p_partkey;
    reader_wrapper = ReaderWrapper(store, target_device,
                                         # name, filepath, threads, cols, filters
                                table_infos= [("part", f"{base_folder}/part.parquet", storage_threads,
                                            ["p_partkey"],None),
                                              ("lineitem", f"{base_folder}/lineitem.parquet", storage_threads,
                                               ["l_partkey"],None)
                                        ],
                                )
    part_reader, lineitem_reader = reader_wrapper.readers
    with torch.cuda.nvtx.range("primes"):
        part_lineitem_hash_init = util.next_prime((scale_factor * 200000) // WORLD_SIZE)
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    part_lineitem = SHUFFLE_MODULE_TYPE(target_device, part_lineitem_hash_init, part_reader.get_fifo(), 0,
                                      lineitem_reader.get_fifo(), 0, store, store_prefix="")
    ######################################################################################
    def tmp_function():
        return part_lineitem(probe_pipeline=None)

    result_str = reader_wrapper.run(tmp_function)
    if iteration == 0:
        print(f"query,iteration,algo,{reader_wrapper.header_str}")
    print(f"line_part_shuffle,{iteration},{part_lineitem.name()},{result_str}")
    ######################################################################################
