import os
import time
import math
from collections import deque

import torch
import torch.nn as nn

import sys
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
from helpers.util import hash_64, divceil

class Distinct(nn.Module):
    def __init__(self, child, unique_col_idx):
        super().__init__()
        self.child = child
        self.unique_col_idx = unique_col_idx

    def get_dtypes(self):
        # Dtypes of output tensors
        return [self.child.get_dtypes()[self.unique_col_idx]]

    def get_morsel_count(self):
        # Morsel count of output tensors
        return self.child.get_morsel_count()

    def forward(self):
        """
        Pull chunks of potentially multiple columns from child,
        and return a one-column chunk of unique values in column unique_col_idx
        """
        torch.cuda.nvtx.range_push("Distinct")

        data = self.child()
        if data is None:
            return None

        return [torch.unique(data[self.unique_col_idx])]


if __name__ == '__main__':
    from modules.morsel_table_module import MorselTableModule

    device = torch.device("cuda:0")
    cols = [torch.arange(0, 4500, dtype=torch.int64, device=device) for i in range(2)]
    cols[0].remainder_(500)
    table = MorselTableModule(cols, 1000)
    tb = Distinct(table,0)
    # tb = Distinct(table,10)
    assert tb.get_morsel_count() == 5, "Wrong morsel count" 

    ######################################################
    while (data := tb()) is not None:
        print(f"{data[0].shape}, {data[0][0].item()} -> {data[0][-1].item()}")
        torch.cuda.synchronize(device=device)

