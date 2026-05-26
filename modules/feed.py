import os
import time
import math
from collections import deque

import torch
import torch.nn as nn

import sys
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))

class Feed(nn.Module):
    def __init__(self, child, gpu):
        super().__init__()
        self.child = child
        self.seen_tuples = 0
        self._gpu = gpu
        self._stream = torch.cuda.Stream(device=self._gpu)

    def forward(self,probe_pipeline=lambda *args, **kwargs: None):
        """
        Feed pipeline one by one
        """
        self.seen_tuples = 0
        torch.cuda.nvtx.range_push("Feed Module forward")

        start = time.perf_counter_ns()
        with torch.cuda.stream(self._stream):
            while (data := self.child()) is not None:
                if len(data) > 0:
                    self.seen_tuples += data[0].shape[0]
                probe_pipeline(data)

        self._stream.synchronize()
        time_ns = time.perf_counter_ns()-start
        torch.cuda.nvtx.range_pop()

        return { "local_result": self.seen_tuples, "time_ns": time_ns,
                "shuffled_bytes": 0}

    def get_stats(self):
        return self.seen_tuples

    def name(self):
        return "Feed"

if __name__ == '__main__':
    from modules.morsel_table_module import MorselTableModule

    device = torch.device("cuda:0")
    cols = [torch.arange(0, 4500, dtype=torch.int64, device=device) for i in range(2)]
    table = MorselTableModule(cols, 1000)
    feed = Feed(table,device)

    ######################################################
    pipeline_fun = lambda data: print(f"Feed got some data of shape {data[0].shape} ")
    print(feed(pipeline_fun))

