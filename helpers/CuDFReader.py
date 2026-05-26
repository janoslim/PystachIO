import time
import torch
import pylibcudf as plc
import pyarrow.parquet as pq
import humanize
import logging
import os
from helpers.FIFO import FIFO, TensorMorsel
from helpers.util import to_torch_dtype, parquet_type_to_torch_dtype

logger = logging.getLogger(__name__)
def logger_setup():
    storage_logging_level = os.environ.get("STORAGE_LOGGING_LEVEL", "INFO").upper()
    logging.basicConfig(level=storage_logging_level, format='%(asctime)s - %(levelname)s: %(message)s')
    logger.info(f"Logging level set to {storage_logging_level}")

class CuDFReader:
    """
    Simple alternative to ParquetFIFOReader, but fully pylibcudf-based.
    """
    
    def __init__(self, parquet_file: str, table_name: str, column_names: list[str]):
        logger_setup()
        self.parquet_file = parquet_file
        self.table_name = table_name
        self.column_names = column_names
        self.dtypes = []
        self._finished = False
        self.start_time = None
        self.end_time = None
        self.prepare_time = 0  # For compatibility with query_wrapper
        self.storage_bytes = {}  # Initialize storage_bytes dictionary
        self.num_threads = 1  # Default to 1 thread for CuDFReader, as it is single-threaded
        self.column_schema = {}
        self.is_decimal = {}
        
        # Read PyArrow metadata for storage calculation
        pq_metadata = pq.read_metadata(self.parquet_file)
        schema = pq_metadata.schema.to_arrow_schema()
        
        # https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ColumnSchema.html
        for col_name in self.column_names:
            col_idx = schema.get_field_index(col_name)
            column_meta = pq_metadata.schema.column(col_idx)
            self.column_schema[col_name] = column_meta
            self.is_decimal[col_name] = column_meta.logical_type.type.startswith("DECIMAL")
            self.dtypes.append(parquet_type_to_torch_dtype(column_meta.physical_type))
            # Get storage bytes and uncompressed size for this column
            col_storage_bytes = 0
            col_uncompressed_bytes = 0
            col_file_actual_bytes = 0
            col_file_uncompressed_bytes = 0
            # For CuDFReader, we read all row groups (no partitioning)
            # Calculate storage for entire file
            for rg_idx in range(pq_metadata.num_row_groups):
                row_group = pq_metadata.row_group(rg_idx)
                rg_column = row_group.column(col_idx)
                assert rg_column.path_in_schema == col_name
                col_storage_bytes += rg_column.total_compressed_size
                col_uncompressed_bytes += self.dtypes[-1].itemsize * row_group.num_rows
                col_file_actual_bytes += rg_column.total_compressed_size
                col_file_uncompressed_bytes += self.dtypes[-1].itemsize * row_group.num_rows
            self.storage_bytes[col_name] = [col_storage_bytes, col_uncompressed_bytes, 0, col_file_actual_bytes, col_file_uncompressed_bytes] # 0 in filtered bytes we can know yet
        
        logger.info(f"[CuDFReader] table={self.table_name} file={self.parquet_file}")
        logger.info(f"             [GLOBAL] total_rowgroups={pq_metadata.num_row_groups} threads={self.num_threads} columns={self.column_names}")
        logger.info(f"             Physical types: {' | '.join([f'{col}:{meta.physical_type}' for col, meta in self.column_schema.items()])}") 
        logger.info(f"             dtypes={self.dtypes}")
        logger.info(f"             Logical types: {' | '.join([f'{col}:{meta.logical_type.type} with {meta.logical_type}'  for col, meta in self.column_schema.items()])}") 
        logger.info(f"             Is Decimal: {' | '.join([f'{col}:{self.is_decimal[col]}' for col in self.column_names ])}") 
        logger.info(f"             Storage Bytes: {' | '.join([f'{col}:{humanize.naturalsize(self.storage_bytes[col][0], binary=True)}(actual):{humanize.naturalsize(self.storage_bytes[col][1], binary=True)}(uncompressed)' for col in self.column_names ])}") 
        logger.info(f"             Total Storage Bytes: {humanize.naturalsize(sum(v[0] for v in self.storage_bytes.values()), binary=True)}(actual):{humanize.naturalsize(sum(v[1] for v in self.storage_bytes.values()), binary=True)}(uncompressed)") 
        logger.info(f"[CuDFReader] [Table {self.table_name}] Simple full-table read mode (no work distribution)")
        
        # Simple FIFO with single producer
        self.fifo = FIFO(
            parent=self,
            num_producers=1,
            table_name=self.table_name,
            columns=self.column_names,
            maxsize=1
        )
        
        logger.info(f"[CuDFReader] Created simple reader for table={self.table_name} file={self.parquet_file}")
        logger.info(f"             columns={self.column_names}")

    def read_and_produce(self):
        try:
            options = plc.io.parquet.ParquetReaderOptions.builder(
                plc.io.SourceInfo([self.parquet_file])
            ).build()
            options.set_columns(self.column_names)
            plc_table_w_meta = plc.io.parquet.read_parquet(options)
            plc_table = plc_table_w_meta.tbl
            
            # Convert columns to tensors (no filtering)
            tensors = []
            for col_idx, col_name in enumerate(self.column_names):
                plc_column = plc_table.columns()[col_idx]
                pytorch_dtype = to_torch_dtype(plc_column.type().id())
                tensor = torch.as_tensor(plc_column.data(), device="cuda").view(pytorch_dtype)
                tensors.append(tensor)
                
                if col_idx == 0:  # Store dtypes on first pass
                    self.dtypes.append(pytorch_dtype)
            
            ##################################
            # if os.environ["TPCH_QUERY"] == "reader":
            #     return # for MULTI_SSD_READER skip the memory stall
            ################################## 
            # Put the result into FIFO
            # self.fifo.put(TensorMorsel(self.column_names, tensors))
        except Exception as e:
            logger.error(f"[CuDFReader] Error reading {self.parquet_file}: {e}")
            raise
        finally:
            self.fifo.producer_finish()
            self.end_time = time.perf_counter_ns()
            self._finished = True

    def start(self):
        if self.start_time is not None:
            return  # Already started
        self.start_time = time.perf_counter_ns()
        self.read_and_produce()
        
    def get_fifo(self) -> FIFO:
        return self.fifo
    
    def get_dtypes(self):
        return self.dtypes
        
    def join(self):
        # Simple implementation - reading is synchronous so nothing to join
        pass
        
    def is_finished(self) -> bool:
        return self._finished

    def set_next_reader(self, next_reader):
        pass
