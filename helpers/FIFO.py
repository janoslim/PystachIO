import queue
import threading
from typing import List, Dict, Iterable
import torch
import os

class TensorMorsel:
    """
       The element in the FIFO, aka a horizontal partition of a table.
       Provides dict-like and list-like access:
      - [int] -> tensor by index, the order is just how you put them
      - get_by_colname("col") -> tensor by column name
      - "col" in batch -> membership test on column names
      - .keys(), .values(), .items()
      - len() -> number of tensors
    """
    __slots__ = ("_columns", "_tensors", "_col2idx")

    def __init__(self, columns: Iterable[str], tensors: Iterable[torch.Tensor]):
        # No type checking here
        self._columns = columns
        self._tensors = tensors
        self._col2idx = {c: i for i, c in enumerate(self._columns)}
    
    def __getitem__(self, index: int):
        """Get tensor by integer index"""
        """  same as: tensor_morsel.tensors()[index]"""
        return self._tensors[index]

    def get_by_colname(self, column_name: str):
        """Get tensor by column name."""
        return self._tensors[self._col2idx[column_name]]

    # TODO:(jigao) this could be deleted...
    def columns(self) -> List[str]:
        return self._columns

    def tensors(self) -> List[torch.Tensor]:
        return self._tensors

    def to_dict(self) -> Dict[str, torch.Tensor]:
        return {c: self._tensors[i] for i, c in enumerate(self._columns)}

    def __len__(self):
        return len(self._tensors)

    def __iter__(self):
        return iter(self._tensors)

    def __contains__(self, key: str):
        return key in self._col2idx

    def items(self):
        for c, t in zip(self._columns, self._tensors):
            yield c, t

    def keys(self):
        return self._columns

    def values(self):
        return self._tensors

    def __repr__(self):
        shapes = ", ".join(f"{c}:{tuple(t.shape)}" for c, t in self.items())
        return f"TensorMorsel({shapes})"

# One FIFO per table, with some columns specified
class FIFO(torch.nn.Module):
    def __init__(self, parent, num_producers: int, table_name: str = "table", columns: List[str] = None, maxsize: int = 0):
        """
        Args:
            table_name: in str
            columns: in str
            maxsize: Maximum queue size (0 = unlimited)
        """
        super().__init__()
        self.parent = parent
        self.table_name = table_name
        if columns is None:
            raise ValueError("'columns' must be provided and cannot be None")
        self.columns: List[str] = list(columns)
        self._queue = queue.Queue(maxsize=maxsize) # This should be the RG-size
        self._finished = threading.Event() # similar to atomic boolean flag
        self._num_producers = num_producers
        self._producers_done = 0
        self._done_lock = threading.Lock()
    
    def put(self, morsel: TensorMorsel):
        """PUT: *blocking* enqueue a TensorMorsel (e.g. horizontal partitioning of a table)."""
        while True:
            try:
                self._queue.put(morsel, block=True, timeout=0.01)
                break
            except queue.Full:
                # Queue is full, continue polling until space is available
                continue

    def forward(self):
        """GET: *blocking* dequeue a RANDOM tensor_list and will wait if empty"""
        while True:
            try:
                morsel = self._queue.get(block=True, timeout=0.01)
                break
            except queue.Empty:
                if self._finished.is_set():
                    # This reader's FIFO is empty, then I can start the next one
                    # This is a decoupled design from query engine
                    sequential_mode = os.environ.get("READER_SEQUENTIAL", "False") == "True"
                    if sequential_mode:
                        self.parent.join()
                    return None
                # Continue polling if not finished
        return morsel

    def producer_finish(self):
        """Signal finish from one thread for one task in this thread"""
        with self._done_lock:
            self._producers_done += 1
            if self._producers_done == self._num_producers:
                self._finished.set()
    
    def empty(self) -> bool:
        """Check if queue is empty."""
        return self._queue.empty()
    
    def qsize(self) -> int:
        """Get approximate queue size."""
        return self._queue.qsize()
    
    def is_finished(self) -> bool:
        """Check if queue is marked as finished."""
        return self._finished.is_set()

    def get_table_name(self) -> str:
        return self.table_name

    def get_columns(self) -> List[str]:
        return self.columns

    def get_dtypes(self):
        return self.parent.get_dtypes()

    def get_morsel_count(self) -> int:
        return self.parent.get_morsel_count()