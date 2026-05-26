import time

import torch
import torch.nn as nn

class Sort(nn.Module):
    def __init__(self, cols):
        super().__init__()
        self.cols = cols

    def forward(self, child):
        data = child()
        sort_cols = [data[col_idx] for col_idx in self.cols]
        sort_cols = torch.stack(sort_cols, dim=1)

        order = sort_cols.argsort(dim=0)
        order = order[:,0]

        return [col[order] for col in data]

class SortPostCombine(nn.Module):
    def __init__(self, project, sort_col_idxs, limit=None):
        super().__init__()
        self.project = project # done before sort_col_idxs are used!
        self.sort_col_idxs = sort_col_idxs
        self.limit = limit
        self.reset()

    def reset(self):
        self.data = []
        self.final_data = []

    def _sort(self, appended_data):
        sort_cols = [appended_data[col_idx] for col_idx in self.sort_col_idxs]
        sort_cols = torch.stack(sort_cols, dim=1)

        order = torch.arange(0,appended_data[0].shape[0],device=appended_data[0].device)
        for col_idx in reversed(range(sort_cols.shape[1])):
            tmp = sort_cols[:, col_idx].argsort(dim=0, stable=(col_idx != sort_cols.shape[1]-1))
            sort_cols = sort_cols[tmp]
            order = order[tmp]

        if self.limit is not None:
            order = order[:self.limit]

        self.final_data = [col[order] for col in appended_data]

    def get_results(self):
        return self.final_data

    def forward(self, data):
        if len(data) == 0 or len(data[0]) == 0:
            return
        if self.project is None:
            self.data.append(data)
        else:
            self.data.append([data[col_idx] for col_idx in self.project])


    def finalize(self,drop_partial=False):
        torch.cuda.synchronize(device=self.data[0][0].get_device())
        appended_data = [torch.cat([chunk[col_idx] for chunk in self.data], dim=0) for col_idx in range(len(self.data[0]))]
        if drop_partial:
            self.data = []

        self._sort(appended_data)




if __name__ == '__main__':
    orig_data = [torch.arange(10), torch.rand(10)]
    print(orig_data)
    s = Sort([1])
    res = s(lambda: orig_data)

    print(res)
