import time
import torch


class ModuloSharding():
    def __init__(self, global_size):
        self._global_size = global_size

    """
    Remainder based sharding for a list of tensor
    """
    def shard_list(self, tensors, index, device):
        mod = torch.empty(tensors[index].shape[0],dtype=torch.int8,device=device)
        torch.remainder(tensors[index], self._global_size, out=mod)
        result = []
        for i in range(self._global_size):
            mask = torch.zeros_like(mod, dtype=torch.int8, device=device)
            mask[mod == i] = 1
            converted_mask = mask.view(-1).nonzero().squeeze()
            entries = torch.bincount(mask)[1]
            cols = [torch.empty((entries, ), dtype=torch.int64, device=device) for _ in range(len(tensors))]
            for j in range(len(tensors)):
                torch.index_select(tensors[j].view(-1), 0, converted_mask, out=cols[j])
            result.append(tuple(cols))
        return tuple(result)

    def shard_list_dynamic(self, tensors, index, device):
        mod = torch.empty(tensors[index].shape[0],dtype=torch.int8,device=device)
        torch.remainder(tensors[index], self._global_size, out=mod)
        result = []
        for i in range(self._global_size):
            mask = torch.zeros_like(mod, dtype=torch.int8, device=device)
            mask[mod == i] = 1
            converted_mask = mask.view(-1).nonzero().squeeze()
            cols = []
            for j in range(len(tensors)):
                # cols.append(tensors[j].view(-1).index_select(0, converted_mask))
                cols.append(torch.index_select(tensors[j], 0, converted_mask))
            result.append(tuple(cols))
        return tuple(result)

    def shard_list_dynamic_mask(self, tensors, index, device):
        mod = torch.empty(tensors[index].shape[0],dtype=torch.int8,device=device)
        torch.remainder(tensors[index], self._global_size, out=mod)
        result = []
        for i in range(self._global_size):
            mask = torch.zeros_like(mod, dtype=torch.bool, device=device)
            mask[mod == i] = True
            cols = []
            for j in range(len(tensors)):
                cols.append(tensors[j].masked_select(mask))
            result.append(tuple(cols))
        return result

    def shard_list_really_nosync(self, tensors, index, device):
        # https://stackoverflow.com/questions/63928630/how-to-retain-2d-or-more-shape-when-using-pytrorch-masked-select
        mod = torch.empty(tensors[index].shape[0],dtype=torch.int8,device=device)
        torch.remainder(tensors[index], self._global_size, out=mod)
        the_sizes = []
        result = []
        for i in range(self._global_size):
            mask = torch.zeros_like(mod, dtype=torch.bool, device=device)
            mask[mod == i] = True
            indices = torch.masked_fill(torch.cumsum(mask.int(),dim=0).sub(1), ~mask, mask.shape[0])
            the_sizes.append(torch.sum(mask))
            cols = []
            for j in range(len(tensors)):
                cols.append(torch.scatter(input=torch.zeros(tensors[j].shape[0]+1,dtype=tensors[j].dtype,device=device), dim=0, index=indices, src=tensors[j]))
            result.append(tuple(cols))
        return result, the_sizes

    def shard_in_place(self, tensors, index, device):
        mod = torch.empty(tensors[index].shape[0],dtype=torch.int8,device=device)
        torch.remainder(tensors[index], self._global_size, out=mod)
        # mod_sorted, order = torch.sort(mod, dim=0)
        # _,input_splits = torch.unique_consecutive(mod_sorted,return_counts=True)
        order = torch.argsort(mod, dim=0)
        input_splits = torch.zeros((self._global_size,),dtype=torch.int64,device=device)
        # unique consecutive has to sync, so we manually count per node part size
        # (scatter_reduce is too slow for this)
        for nodei in range(self._global_size):
            input_splits[nodei] = (mod == nodei).sum()

        # print("original data:",tensors)
        # print("mod:",mod)
        # print("mod_sorted:",mod_sorted)
        # print("input_splits:",input_splits)
        # print("order:",order)
        # reordered = tensors[index][order]
        # print("reordered:",reordered)
        result = []
        for col in tensors:
            result.append(col[order])
            # result.append(torch.index_select(col, 0, order))
        return result,input_splits


    def shard_list_dynamic_where(self, tensors, index, device):
        mod = torch.empty(tensors[index].shape[0],dtype=torch.int8,device=device)
        torch.remainder(tensors[index], self._global_size, out=mod)
        result = []
        for i in range(self._global_size):
            mask = torch.zeros_like(mod, dtype=torch.int8, device=device)
            mask[mod == i] = 1
            converted_mask = torch.where(mask)[0]
            cols = []
            for j in range(len(tensors)):
                cols.append(tensors[j].index_select(0, converted_mask))
            result.append(tuple(cols))
        return tuple(result)

