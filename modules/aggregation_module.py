import sys
import os
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))

import torch
import torch.nn as nn

from helpers.util import cub_unique_consecutive

def group_by(data, gb_col_idxs, preformat=False, projection=None, use_consecutive_counts=False):
    """
    :param data: list of table columns
    :param gb_col_idxs: indexes of columns in data to group by
    :param preformat
    :param projection: indexes of columns for projection (after group by)

    """
    torch.cuda.nvtx.range_push("Group By")
    if preformat:
        # data[0] = groups
        # data[1] = results
        gb_cols = data[0]
    else:
        gb_cols = [data[col_idx] for col_idx in gb_col_idxs]
        gb_cols = torch.stack(gb_cols, dim=1)

    order = torch.arange(0,data[0].shape[0],device=data[0].device)
    # print("order:",order)
    # print("gb_cols:",gb_cols)

    with torch.cuda.nvtx.range("Order"):
        for col_idx in reversed(range(gb_cols.shape[1])):
            tmp = gb_cols[:, col_idx].argsort(dim=0, stable=(col_idx != gb_cols.shape[1]-1))
            gb_cols = gb_cols[tmp]
            order = order[tmp]

    # print("Order: ",order)
    # print("sorted gb_cols:",gb_cols)

    with torch.cuda.nvtx.range("Sorting data"):
        if projection is None:
            # order all columns
            sorted_data = [col[order] for col in data]
        else:
            sorted_data = [data[colidx][order] for colidx in projection]
    # print("sorted_data:",sorted_data)

    with torch.cuda.nvtx.range("Unique consecutive"):
        if use_consecutive_counts:
            groups_unique, group_idx = torch.unique_consecutive(gb_cols, return_inverse=False, return_counts=True, dim=0)
        else:
            groups_unique, group_idx = torch.unique_consecutive(gb_cols, return_inverse=True, return_counts=False, dim=0)
        # not necessary anymore, since unique_consecutive got fixed
        # groups_unique, group_idx = cub_unique_consecutive(gb_cols, return_inverse=True, dim=0)
    # print("groups_unique:",groups_unique)
    # print("group_idx:",group_idx)

    # now data is sorted correctly, group_idx can be used as a mapping from data row index
    # to index in group_unique, i.e. to the group values

    torch.cuda.nvtx.range_pop()
    return sorted_data, groups_unique, group_idx


class Aggregation(nn.Module):
    def __init__(self, agg_fun, combine_agg, gb_col_idxs):
        super().__init__()
        self.agg_fun = agg_fun
        self.combine_agg = combine_agg
        self.gb_col_idxs = gb_col_idxs
        self.res = []

    def forward(self, data):
        sorted_data, groups_unique, group_idx = group_by(data,self.gb_col_idxs)
        results = self.agg_fun(sorted_data, groups_unique, group_idx)
        if not self.res:
            self.res.append(groups_unique)
            self.res.append(results)
        else:
            appended_groups = torch.cat([self.res[0],groups_unique], dim=0)
            # print("appended_groups:\n",appended_groups)
            appended_results = torch.cat([self.res[1],results], dim=0)
            # print("appended_results:\n",appended_results)
            sorted_data, groups_unique, group_idx = group_by([appended_groups,appended_results],[],preformat=True)
            results = self.combine_agg(sorted_data, groups_unique, group_idx)
            self.res = [groups_unique, results]
            # print("Existing results")
        return self.res


class AggregationPostCombine(nn.Module):
    def __init__(self, agg_fun, combine_agg, gb_col_idxs, projection=None, use_consecutive_counts=False):
        super().__init__()
        self.agg_fun = agg_fun
        self.combine_agg = combine_agg
        self.gb_col_idxs = gb_col_idxs
        self.projection = projection
        self.use_consecutive_counts = use_consecutive_counts
        self.reset()

    def reset(self):
        self.groups = []
        self.ress = []
        self.final_groups = None
        self.final_aggs = None

    def get_results(self):
        return (self.final_groups, self.final_aggs)

    def forward(self, data):
        sorted_data, groups_unique, group_idx = group_by(data,self.gb_col_idxs,projection=self.projection,use_consecutive_counts=self.use_consecutive_counts)
        with torch.cuda.nvtx.range("Agg fun"):
            results = self.agg_fun(sorted_data, groups_unique, group_idx)

        self.groups.append(groups_unique)
        self.ress.append(results)

    def finalize(self,drop_partial=False):
        torch.cuda.synchronize(device=self.groups[0].get_device())
        # print(f"After synchronize in AggregationPostCombine")
        appended_groups = torch.cat(self.groups, dim=0)
        appended_results = torch.cat(self.ress, dim=0)
        sorted_data, groups_unique, group_idx = group_by([appended_groups,appended_results],[],preformat=True,use_consecutive_counts=self.use_consecutive_counts)
        results = self.combine_agg(sorted_data, groups_unique, group_idx)
        self.final_groups = groups_unique
        self.final_aggs = results
        if drop_partial:
            torch.cuda.synchronize(device=self.groups[0].get_device())
            self.groups = []
            self.ress = []


####################################################################################################
#Helpers
# Simple aggregation function, summing a given column
def create_scatter_agg_fun(col_idx,reduce="sum"):
    def query_agg(sorted_data,groups_unique,group_idx):
        results = torch.zeros(groups_unique.shape[0],dtype=sorted_data[col_idx].dtype,device=sorted_data[col_idx].get_device()).scatter_reduce(0, group_idx, sorted_data[col_idx], reduce=reduce)
        return results
    return query_agg

def create_counts_agg_fun(col_idx,reduce="sum"):
    def query_agg(sorted_data,groups_unique,group_counts):
        num_groups = groups_unique.shape[0]
        results = torch.zeros(num_groups,dtype=sorted_data[col_idx].dtype,device=sorted_data[col_idx].get_device())
        prefix_sum = torch.zeros(num_groups+1, dtype=torch.int64, device="cuda:0")
        sums = torch.zeros(num_groups,dtype=torch.int64,device=sorted_data[col_idx].get_device())
        torch.cumsum(group_counts, dim=0, out=prefix_sum[1:])
        for i in range(num_groups):
            results[i] = sorted_data[col_idx][prefix_sum[i]:prefix_sum[i+1]].sum()

        return results
    return query_agg
####################################################################################################

if __name__ == '__main__':
    device = torch.device("cuda:0")
    data_len = 20
    # num_groups = 12
    # data = [torch.arange(0+data_len*col_idx, data_len*(col_idx+1), 1, dtype=torch.int64, device=device).remainder(num_groups) for col_idx in range(2)] + [torch.rand(data_len,device=device) for col_idx in range(2)]
    data =    [ torch.tensor([0,1,2,3,4,5,6,7,8,9,9,7,8,6,5,4,3,2,1,0], dtype=torch.int64, device=device)] + [torch.tensor([1,2,3,4,5,6,7,8,9,9,8,7,8,6,5,4,3,2,1,0], dtype=torch.int64, device=device)] + [torch.rand(data_len,device=device) for col_idx in range(2)]
    print("Data 0:",[f'{item:>6.2f}' for item in data[0].tolist()])
    print("Data 1:",[f'{item:>6.2f}' for item in data[1].tolist()])
    print("Data 2:",[f'{item:>6.2f}' for item in data[2].tolist()])
    print("Data 3:",[f'{item:>6.2f}' for item in data[3].tolist()])

    # torch.cuda.set_sync_debug_mode(1)
    # Select 0,1,mean(3-2) from data
    def agg_fun(sorted_data,groups_unique,group_idx):
        diff = sorted_data[3] - sorted_data[2]
        results = torch.zeros(groups_unique.shape[0],dtype=diff.dtype,device=device).scatter_reduce(0, group_idx, diff, reduce="mean")
        return results

    # Select 0,1,sum(3+2) from data
    def agg_fun2(sorted_data,groups_unique,group_idx):
        diff = sorted_data[3] + sorted_data[2]
        results = torch.zeros(groups_unique.shape[0],dtype=diff.dtype,device=device).scatter_reduce(0, group_idx, diff, reduce="sum")
        return results

    def agg_fun3(sorted_data,groups_unique,group_idx):
        results = torch.zeros((groups_unique.shape[0],2),dtype=sorted_data[2].dtype,device=device)
        results[:,0].scatter_reduce_(0, group_idx, sorted_data[2], reduce="sum")
        results[:,1].scatter_reduce_(0, group_idx, sorted_data[3], reduce="sum")
        # results = torch.zeros((groups_unique.shape[0],2),dtype=sorted_data[2].dtype,device=device).scatter_reduce(0, group_idx, sorted_data[2], reduce="sum")
        # results.scatter_reduce_()
        return results

    def agg_fun4(sorted_data,groups_unique,group_idx):
        results = torch.zeros((groups_unique.shape[0], 2),dtype=sorted_data[0].dtype,device=device)
        results[:,0].scatter_reduce_(0, group_idx, sorted_data[0], reduce="sum")
        results[:,1].scatter_reduce_(0, group_idx, sorted_data[1], reduce="sum")
        return results

    def combine_agg(sorted_data,groups_unique,group_idx):
        # print("sorted_data:\n",sorted_data)
        results = torch.zeros(groups_unique.shape[0],dtype=sorted_data[1].dtype,device=device).scatter_reduce(0, group_idx, sorted_data[1], reduce="sum")
        return results

    def combine_agg_multi(sorted_data,groups_unique,group_idx):
        results = torch.zeros((groups_unique.shape[0],2),dtype=sorted_data[1].dtype,device=sorted_data[0].get_device())

        # print(f"{results.dtype}, {groups_unique.dtype}, {sorted_data[1][].dtype}, {sorted_data[2].dtype}, {group_idx.dtype}")

        # copy1 = sorted_data[1].clone().detach()
        # copy2 = sorted_data.clone().detach()
        results[:,0].scatter_reduce_(0, group_idx, sorted_data[1][:,0], reduce="sum")
        results[:,1].scatter_reduce_(0, group_idx, sorted_data[1][:,1], reduce="sum")
        return results

    # warm up:
    sorted_data,groups_unique,group_idx = group_by(data, [0,1])
    results = agg_fun(sorted_data,groups_unique,group_idx)
    torch.cuda.synchronize()

    # manual way:
    with torch.torch.cuda.nvtx.range("Single Aggregation"):
        sorted_data,groups_unique,group_idx = group_by(data, [0,1])
        results = agg_fun(sorted_data,groups_unique,group_idx)
    torch.cuda.synchronize()
    # print("Results:", results)

    # agg_module = Aggregation(agg_fun2, combine_agg, [0,1])
    # agg_module(data)
    # torch.cuda.set_sync_debug_mode(0)
    # print("after first aggregation: ", agg_module.res)
    # agg_module(data)
    # print("after second aggregation: ", agg_module.res)

    with torch.torch.cuda.nvtx.range("Multiple Aggregation"):
        # agg_fun3 and agg_fun4 are the same with and without projection
        # agg_module = AggregationPostCombine(agg_fun3, combine_agg, [0,1])
        agg_module = AggregationPostCombine(agg_fun4, combine_agg_multi, [0,1], projection=[2,3])
        agg_module(data)
        agg_module.finalize()
    torch.cuda.synchronize()
    print("after first aggregation: ", agg_module.groups[0])
    print(agg_module.ress[0])


