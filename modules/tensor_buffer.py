import os
import time
import math
from collections import deque

import torch
import torch.nn as nn

import sys
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
from helpers.util import hash_64, divceil

class TensorBuffer(nn.Module):
    def __init__(self, child, concat_num):
        super().__init__()
        self.child = child
        self.child_done = False
        self.concat_num = concat_num
        if self.concat_num < 1:
            print(f"WARN: TensorBuffer concat_num is not valid, should be >= 1, is {self.concat_num }")

    def get_dtypes(self):
        # Dtypes of output tensors
        return self.child.get_dtypes()

    def get_morsel_count(self):
        # Morsel count of output tensors
        return divceil(self.child.get_morsel_count(), self.concat_num)

    def forward(self):
        """
        Call child concat_num times then pass on
        """
        if self.child_done:
            return None

        tensorss = [] # list of list of tensors (columns)
        torch.cuda.nvtx.range_push("TensorBuffer")

        for i in range(self.concat_num):
            data = self.child()
            if data is None:
                self.child_done = True
                break

            tensorss.append(data)

        if len(tensorss) == 0:
            torch.cuda.nvtx.range_pop()
            return None

        result = [torch.cat(tuple(chunk[col_idx] for chunk in tensorss)) for col_idx in range(len(tensorss[0]))]
        torch.cuda.nvtx.range_pop()

        return result


if __name__ == '__main__':
    from modules.morsel_table_module import MorselTableModule

    device = torch.device("cuda:0")
    cols = [torch.arange(0, 4500, dtype=torch.int64, device=device) for i in range(2)]
    table = MorselTableModule(cols, 1000)
    tb = TensorBuffer(table,3)
    # tb = TensorBuffer(table,10)
    assert tb.get_morsel_count() == 2, "Wrong morsel count" 

    ######################################################
    while (data := tb()) is not None:
        print(f"{data[0].shape}, {data[0][0].item()} -> {data[0][-1].item()}")
        torch.cuda.synchronize(device=device)

