import os
import time
import math
from collections import deque

import torch
import torch.nn as nn

import sys
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
from helpers.util import hash_64

class PipelineBreaker(nn.Module):
    def __init__(self, child):
        super().__init__()
        self.child = child
        self.child_done = False
        self.tensors = deque()

    def get_dtypes(self):
        # Dtypes of output tensors
        return self.child.get_dtypes()

    def get_morsel_count(self):
        # Morsel count of output tensors
        return self.child.get_morsel_count()

    def forward(self):
        """
        First forward drains the child iterator fully, then dispenses
        """
        if self.child_done:
            if self.tensors:
                return self.tensors.popleft()
            else:
                return None

        # child not yet drained

        torch.cuda.nvtx.range_push("PB drain child")

        while (data := self.child()) is not None:
            self.tensors.append(data)

        self.child_done = True
        torch.cuda.nvtx.range_pop()

        if self.tensors:
            return self.tensors.popleft()
        else:
            return None


if __name__ == '__main__':
    from modules.morsel_table_module import MorselTableModule

    device = torch.device("cuda:0")
    cols = [torch.arange(0, 4500, dtype=torch.int64, device=device) for i in range(2)]
    table = MorselTableModule(cols, 1000)
    pb = PipelineBreaker(table)

    ######################################################
    while (data := pb()) is not None:
        print(f"{data[0].shape}")
        torch.cuda.synchronize(device=device)

