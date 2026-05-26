import torch
import torch.nn as nn
import math


class MorselTableModule(nn.Module):
    def __init__(self, columns, tuples_per_morsel, full_repeat=1):
        super().__init__()
        self.columns = columns
        self.tuples_per_morsel = tuples_per_morsel
        self.sent = 0
        self.full_sent = 0
        self.full_repeat = full_repeat

    def bytes(self):
        return sum([x.element_size() * x.nelement() for x in self.columns])*self.full_repeat

    def forward(self):
        if self.sent >= len(self.columns[0]):
            self.full_sent += 1
            if self.full_sent >= self.full_repeat:
                return None
            self.sent = 0

        to_send = self.tuples_per_morsel
        if self.sent + self.tuples_per_morsel > len(self.columns[0]):
            to_send = len(self.columns[0]) - self.sent
        data = [t[self.sent:self.sent + to_send] for t in self.columns]
        self.sent += to_send
        return data

    def get_dtypes(self):
        return [col.dtype for col in self.columns]

    def get_row_count(self):
        return len(self.columns[0])*self.full_repeat

    def get_morsel_count(self):
        return int(math.ceil(len(self.columns[0]) / self.tuples_per_morsel))*self.full_repeat

    def get_col_count(self):
        return len(self.columns)

    def reset(self):
        self.sent = 0
        self.full_sent = 0


if __name__ == '__main__':
    test_cols = [torch.arange(0,4500,dtype=torch.int64, device="cpu") for i in range(2)]
    test_table = MorselTableModule(test_cols, 1000, full_repeat=1)

    while (data := test_table()):
        if data[0] is None:
            break
        print(data[0].shape)