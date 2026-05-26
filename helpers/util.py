import math
import torch
import datetime

import cupy as cp
import numpy as np
from sympy import isprime
from pylibcudf.types import TypeId

def data_creation(target_device, num_cols, world_rank, world_size, total_target, zipf):
    zipf_num = [0] + zipf_numbers(world_size, zipf, total_target)
    zipf_prefix = np.cumsum(zipf_num)
    # e.g. [0, 313, 254, 225, 208]
    # print(zipf_prefix)
    # print(f"# This node will have {zipf_num[world_rank+1]} of {total_target} tuples (ratio {zipf_num[world_rank+1]/ total_target:.6f})")
    # print(f"# from range [{zipf_prefix[world_rank]}, {zipf_prefix[world_rank+1]}]")

    kwargs = {}
    if target_device == torch.device("cpu") or target_device == "cpu":
        kwargs["pin_memory"] = True

    data = [torch.arange(zipf_prefix[world_rank] + i * 1000000000,
                              zipf_prefix[world_rank+1] + i * 1000000000,
                              dtype=torch.int64, device=target_device,**kwargs) for i in range(num_cols)]
    return data


def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB", "ZiB", "YiB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])


def date_to_epoch_days(year,month,day):
    return (datetime.datetime(year,month,day) - datetime.datetime(1970,1,1)).days

def epoch_days_to_datetime(epoch_days):
    return (datetime.datetime(1970,1,1) + datetime.timedelta(days=epoch_days))

def next_prime(number):
    while not isprime(number):
        number += 1
    return number

def divceil(a,b):
    return -(a // -b)

def hash_64(tensor, div):
    # simple remainder
    result = torch.empty_like(tensor,dtype=torch.int64)
    torch.remainder(tensor, div, out=result)
    return result

def chunked_unique(buffer_list, index_col):
    """
    Process unique values in chunks to avoid OOM.
    
    Args:
        buffer_list: List of buffers containing data
        index_col: Index of the column to extract unique values from
        
    Returns:
        Sorted unique values tensor
    """
    unique_parts = []
    for buffer in buffer_list:
        col = buffer[index_col]
        unique_parts.append(torch.unique(col, sorted=False)) # pre unique
    index_col_concat = torch.cat(unique_parts)
    distinct_vals = torch.unique(index_col_concat, sorted=False) # after concatenating, the final unique
    return distinct_vals

def zipf_seq(N,s):
    res = [0]*N
    denom = 0.0
    for n in range(1,N+1):
        denom += 1.0/(n**s)
    for k in range(1,N+1):
        res[k-1] = (1.0/(k**s))/denom
    return res

def zipf_numbers(N,s,total):
    seq = zipf_seq(N,s)
    res = [int(d * total) for d in seq]
    res[-1] += total - sum(res)
    assert sum(res) == total
    return res

def parquet_type_to_torch_dtype(physical_type_str: str) -> torch.dtype:
    type_mapping = {
        'BOOLEAN': torch.bool,
        'INT32': torch.int32,
        'INT64': torch.int64,
        'FLOAT': torch.float32,
        'DOUBLE': torch.float64,
        'BYTE_ARRAY': torch.uint8,  # For string/binary data
        'FIXED_LEN_BYTE_ARRAY': torch.uint8,  # For fixed-length binary
        'INT96': torch.int64,  # Legacy timestamp format, treat as int64
    }
    if physical_type_str in type_mapping:
        return type_mapping[physical_type_str]
    else:
        raise NotImplementedError(f"Unsupported Parquet physical type: {physical_type_str}")

def to_torch_dtype(type_id: TypeId) -> torch.dtype:
    if type_id == TypeId.INT8:   return torch.int8
    if type_id == TypeId.INT16:  return torch.int16
    if type_id == TypeId.INT32:  return torch.int32
    if type_id == TypeId.INT64:  return torch.int64
    if type_id == TypeId.UINT8:  return torch.uint8
    if type_id == TypeId.FLOAT32:return torch.float32
    if type_id == TypeId.FLOAT64:return torch.float64
    if type_id == TypeId.BOOL8:  return torch.bool

    if type_id in (
        TypeId.TIMESTAMP_DAYS,
    ):
        return torch.int32
    
    # Decimals: expose underlying storage
    if type_id == TypeId.DECIMAL32:  return torch.int32
    if type_id == TypeId.DECIMAL64:  return torch.int64
    if type_id == TypeId.DECIMAL128: raise NotImplementedError("DECIMAL128 not supported (no 128-bit int in torch)")
    
    raise NotImplementedError(f"Unhandled TypeId {type_id.name}")

def print_parquet_metadata(file):
    print("\n=== PARQUET FILE METADATA ===", flush=True)
    metadata = pq.read_metadata(file)
    for i in range(metadata.num_columns):
        column_meta = metadata.schema.column(i)
        print(f"Column {i} {column_meta.name}: physical_type: {column_meta.physical_type}, logical_type: {column_meta.logical_type}")
    #table = pq.read_table(file)
    #for col in table.schema:
    #    print(f"{col.name}: {col.type}") # this is the arrow schema datatype, not the Parquet datatype
    # plc_metadata = plc.io.parquet_metadata.read_parquet_metadata(plc.io.SourceInfo([file]))
    # print(plc_metadata.schema())
    # print(plc_metadata.metadata())

def check_duplicates_in_cudf_column(plc_column, col_name, rg, thread_id, table_name):
    """
    Simple helper function to check for duplicates in cuDF column using CPU/NumPy
    Call this before converting to PyTorch tensor to isolate duplicate sources
    """
    try:
        # Convert to CPU numpy array for safe duplicate checking
        cupy_array = cp.asarray(plc_column.data())
        if len(cupy_array) <= 1:
            return
            
        pytorch_dtype = to_torch_dtype(plc_column.type().id())
        if pytorch_dtype == torch.int32:
            numpy_dtype = np.int32
        elif pytorch_dtype == torch.int64:
            numpy_dtype = np.int64
        elif pytorch_dtype == torch.int16:
            numpy_dtype = np.int16
        elif pytorch_dtype == torch.int8:
            numpy_dtype = np.int8
        elif pytorch_dtype == torch.uint8:
            numpy_dtype = np.uint8
        elif pytorch_dtype == torch.float32:
            numpy_dtype = np.float32
        elif pytorch_dtype == torch.float64:
            numpy_dtype = np.float64
        elif pytorch_dtype == torch.bool:
            numpy_dtype = np.bool_
        else:
            numpy_dtype = np.int64  # fallback
        cupy_viewed = cupy_array.view(numpy_dtype)
        numpy_array = cupy_viewed.get().astype(np.int64)
        
        # Simple duplicate check using basic operations 
        unique_array = np.unique(numpy_array)
        if len(unique_array) < len(numpy_array):
            values, counts = np.unique(numpy_array, return_counts=True)
            duplicate_mask = counts > 1
            if np.any(duplicate_mask):
                dup_values = values[duplicate_mask]
                dup_counts = counts[duplicate_mask]
                dup_summary = ", ".join([f"{val}({count}x)" for val, count in zip(dup_values.tolist(), dup_counts.tolist())])
                logger.info(f"!!![CPU_DUP_CHECK] [Producer{thread_id}] [Table {table_name}] RG{rg} col_name={col_name} | Original:{len(numpy_array)} Unique:{len(unique_array)} | Duplicates: {dup_summary}")
    except Exception as e:
        logger.info(f"!!![CPU_DUP_CHECK] Error for RG{rg} col_name={col_name}: {e}")

# FROM CCCL TEST
class CCCL_Stream:
    """
    Simple cupy stream wrapper that implements the __cuda_stream__ protocol.
    """

    def __init__(self, cp_stream):
        self.cp_stream = cp_stream

    def __cuda_stream__(self):
        return (0, self.cp_stream.ptr)

    @property
    def ptr(self):
        return self.cp_stream.ptr


def cub_unique_consecutive(data, **kwargs):
    n = data.shape[0]

    @parallel.gpu_struct
    class Row:
        first: np.int64
        second: np.int64

    torch_stream = torch.cuda.current_stream()
    cupy_stream = cp.cuda.ExternalStream(torch_stream.cuda_stream)
    cccl_stream = CCCL_Stream(cupy_stream)

    with cupy_stream:
        # unique_by_key
        d_in_keys = cp.asarray(data).view(Row.dtype)
        d_in_values = cp.arange(n, dtype=cp.int64)  # Use indices as values
        d_out_keys = cp.empty_like(d_in_keys)
        d_out_values = cp.empty_like(d_in_values)
        d_out_num_selected = cp.empty(1, dtype=cp.int64)
        pinned_mem = cp.cuda.alloc_pinned_memory(8)
        num_selected_host = np.frombuffer(pinned_mem, dtype=d_out_num_selected.dtype)
        def mycompare(left_val,right_val):
            return left_val.first == right_val.first and left_val.second == right_val.second
        parallel.unique_by_key(
            d_in_keys,
            d_in_values,
            d_out_keys,
            d_out_values,
            d_out_num_selected,
            # parallel.OpKind.EQUAL_TO,
            mycompare,
            n,
            cccl_stream
        )
        cp.cuda.runtime.memcpyAsync(num_selected_host.ctypes.data,
                                    d_out_num_selected.data.ptr, 8, cp.cuda.runtime.memcpyDeviceToHost, cupy_stream.ptr)
        torch_stream.synchronize()
        num_selected = num_selected_host[0]
        unique_values = d_out_keys[:num_selected]
        first_indices = d_out_values[:num_selected]

        # CUB scan for inverse indices
        changes = cp.zeros(n, dtype=cp.int64)
        changes[first_indices] = 1 # bit set the changes
        inverse = cp.empty(n, dtype=cp.int64)
        parallel.inclusive_scan(changes, inverse, parallel.OpKind.PLUS, np.array(0, dtype=np.int64), n, cccl_stream)
        inverse = inverse - 1

    return torch.as_tensor(unique_values.view(np.int64)), torch.as_tensor(inverse)


if __name__ == '__main__':
    import time

    # start = time.perf_counter_ns()
    # tmp = next_prime(225000000)
    # end = time.perf_counter_ns()
    # print(f"{tmp}, took {(end-start)/1e9} seconds to find")
    # data = torch.tensor([42,42,42,42,42, 10, 10, 20, 20, 20, 30, 100, 100], dtype=torch.int64, device="cuda:0")
    data = torch.tensor([[2,2],[2,2],[2,1],[2,1],[0,1],[0,0],[0,0],[0,0]], dtype=torch.int64, device="cuda:0")
    
    # CUB version
    unique_cub, inverse_cub = cub_unique_consecutive(data, return_inverse=True, dim=0)
    
    # PyTorch version
    unique_torch, inverse_torch = torch.unique_consecutive(data, return_inverse=True, dim=0)
    
    print(f"Input: {data}")
    print(f"CUB unique: {unique_cub}")
    print(f"PyTorch unique: {unique_torch}")
    print(f"CUB inverse: {inverse_cub}")
    print(f"PyTorch inverse: {inverse_torch}")
