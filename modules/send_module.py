import torch
import torch.distributed as dist

def send_with_store(store, store_prefix, world_rank, world_size, iteration, data, dtypes, target_device=0):
    """
        Shuffle of partitioned data (= [Data for Node0=[Col0,Col1, ...],
                                        ...,
                                        Data for Node0=[Col0,Col1, ...],])
        Uses the given store to distribute send/recv size information
        Information in store:
        "{store_prefix}_size_{iteration}_{world_rank}_to_{i}" -> "{data_size}"
        e.g. if in iteration 1, node 5 shuffles 5000 tuples to node 2:
        "some_prefix_size_1_5_to_2" -> "5000"
    """

    num_cols = len(dtypes)

    recv_data = [[None for _ in range(num_cols)] for _ in range(world_size)]
    length_buffers = [-1 for _ in range(world_size)]
    for i in range(world_size):
        if i != world_rank:
            data_size = data[i][0].shape[0]
            store.set(f"{store_prefix}_size_{iteration}_{world_rank}_to_{i}", str(data_size))
        else:
            length_buffers[i] = data[i][0].shape[0]
    for i in range(world_size):
        if i != world_rank:
            key = f"{store_prefix}_size_{iteration}_{i}_to_{world_rank}"
            store.wait([key])
            length_buffers[i] = int(store.get(key))

    for c in range(num_cols):
        for i in range(world_size):
            recv_data[i][c] = torch.empty((length_buffers[i]), dtype=dtypes[c],device=target_device)
            sender_data_list = [data[i][c] for i in range(world_size)]

        recv_list = [recv_data[i][c] for i in range(world_size)]
        dist.all_to_all(input_tensor_list=sender_data_list, output_tensor_list=recv_list)
    return recv_data


def send_with_store_var_cols(store, store_prefix, world_rank, world_size, iteration, data, dtypes, target_device=0):
    """
        Shuffle of partitioned data (= [Data for Node0=[Col0,Col1, ...],
                                        ...,
                                        Data for Node0=[Col0,Col1, ...],])
        Uses the given store to distribute send/recv size information.
        This variant allows to shuffle columns of variable lengths, i.e. for compressed column data.
        Information in store:
        "{store_prefix}_size_{iteration}_{world_rank}_col_{col}_to_{i}" -> "{data_size}"
        e.g. if in iteration 1, node 5 shuffles 5000 tuples for column 16 to node 2:
        "some_prefix_size_1_5_to_2" -> "5000"
    """

    num_cols = len(dtypes)

    recv_data = [[None for _ in range(num_cols)] for _ in range(world_size)]
    length_buffers = [[-1 for _ in range(num_cols)] for _ in range(world_size)]
    has_local = len(data) > 0
    for i in range(world_size):
        for c in range(num_cols):
            if i != world_rank:
                if not has_local:
                    data_size = -1
                else:
                    data_size = data[i][c].size()[0]
                store.set(f"{store_prefix}_size_{iteration}_{world_rank}_col_{c}_to_{i}", str(data_size))
            else:
                if has_local:
                    recv_data[i][c] = data[i][c]
                else:
                    recv_data[i][c] = None
    for i in range(world_size):
        if i == world_rank:
            continue
        for c in range(num_cols):
            key = f"{store_prefix}_size_{iteration}_{i}_col_{c}_to_{world_rank}"
            store.wait([key])
            length_buffers[i][c] = int(store.get(key))

    for c in range(num_cols):
        for i in range(world_size):
            if length_buffers[i][c] > 0:
                recv_data[i][c] = torch.empty((length_buffers[i][c]), dtype=dtypes[c], device=target_device)
        if len(data) > 0:
            sender_data_list = [data[i][c] for i in range(world_size)]
        else:
            sender_data_list = [torch.empty((0,), dtype=dtypes[c], device=target_device) for i in range(world_size)]
        recv_list = [recv_data[i][c] if recv_data[i][c] is not None
                        else torch.empty((0,), dtype=dtypes[c], device=target_device)
                            for i in range(world_size)]
        dist.all_to_all(input_tensor_list=sender_data_list, output_tensor_list=recv_list)
    return recv_data


def send_with_ccl(world_size, data, dtypes, target_device=0, group=None):
    """
        Shuffle of partitioned data (= [Data for Node0=[Col0,Col1, ...],
                                        ...,
                                        Data for NodeN=[Col0,Col1, ...],])
        Uses collectives to distribute send/recv size information
        Returns                        [Data from Node0=[Col0,Col1, ...],
                                        ...,
                                        Data from NodeN=[Col0,Col1, ...],]
    """
    num_cols = len(dtypes)

    recv_data = [[None for _ in range(num_cols)] for _ in range(world_size)]
    send_sizes = torch.tensor([node_data[0].shape[0] for node_data in data], dtype=torch.int64, device="cpu")
    recv_sizes = torch.empty(world_size, dtype=torch.int64, device="cpu")
    dist.all_to_all_single(input=send_sizes, output=recv_sizes, group=group)

    for c in range(num_cols):
        for i in range(world_size):
            recv_data[i][c] = torch.empty(recv_sizes[i], dtype=dtypes[c], device=target_device)

        sender_data_list    = [data[i][c] for i in range(world_size)]
        recv_list           = [recv_data[i][c] for i in range(world_size)]

        dist.all_to_all(input_tensor_list=sender_data_list, output_tensor_list=recv_list, group=group)
    return recv_data


def send_with_ccl_combine_recv(world_size, data, dtypes, target_device=0, group=None):
    """
        Shuffle of partitioned data (= [Data for Node0=[Col0,Col1, ...],
                                        ...,
                                        Data for Node0=[Col0,Col1, ...],])
        Uses collectives to distribute send/recv size information
        Returns                         [Col0,Col1, ...]
    """
    num_cols = len(dtypes)

    send_sizes = torch.tensor([node_data[0].shape[0] for node_data in data], dtype=torch.int64, device="cpu")
    recv_sizes = torch.empty(world_size, dtype=torch.int64, device="cpu")
    dist.all_to_all_single(input=send_sizes, output=recv_sizes, group=group)

    prefix_sum = torch.zeros(world_size+1, dtype=torch.int64, device="cpu")
    # print(f"{recv_sizes=}")
    torch.cumsum(recv_sizes, dim=0, out=prefix_sum[1:])
    # print(f"{prefix_sum=}")
    total_tuples = prefix_sum[-1].item()
    recv_data = [torch.empty(total_tuples, dtype=dtypes[c], device=target_device) for c in range(num_cols)]

    for c in range(num_cols):
        sender_data_list    = [data[i][c] for i in range(world_size)]
        recv_list           = [recv_data[c][prefix_sum[i]:prefix_sum[i+1]] for i in range(world_size)]

        dist.all_to_all(input_tensor_list=sender_data_list, output_tensor_list=recv_list, group=group)
    return recv_data



def send_inplace_with_ccl(world_size, data, input_splits, dtypes, target_device=0, group=None):
    """
        Shuffle of  data (= [Col0, Col1])
        Input splits defines which parts of the inplace partitioned columns goes to which node, i.e.
        input_splits = [5,6,2] means the first 5 data items go to node0, the next 6 to node1, the next 2 to node2 
        Uses collectives to distribute send/recv size information
        Returns                         [Col0,Col1, ...]
    """
    num_cols = len(dtypes)

    recv_sizes = torch.empty((world_size,), dtype=torch.int64, device="cpu")
    input_splits_cpu = torch.empty((world_size,), dtype=torch.int64, device="cpu", pin_memory=True)
    input_splits_cpu.copy_(input_splits, non_blocking=True)
    torch.cuda.current_stream().synchronize()

    dist.all_to_all_single(input=input_splits_cpu, output=recv_sizes, group=group)

    input_splits_list = list(input_splits_cpu)
    recv_sizes_list = list(recv_sizes)

    total_received = recv_sizes.sum().item()

    recv_data = [torch.empty((total_received,), dtype=dtypes[c], device=target_device) for c in range(num_cols)]
    for c in range(num_cols):
        dist.all_to_all_single(input=data[c], output=recv_data[c],
                input_split_sizes=input_splits_list, output_split_sizes=recv_sizes_list,
                group=group)

    return recv_data


def send_inplace_with_ccl_per_rank(world_size, data, input_splits, dtypes, target_device=0, group=None):
    """
        Shuffle of  data (= [Col0, Col1])
        Input splits defines which parts of the inplace partitioned columns goes to which node, i.e.
        input_splits = [5,6,2] means the first 5 data items go to node0, the next 6 to node1, the next 2 to node2 
        Uses collectives to distribute send/recv size information
        Returns                        [Data from Node0=[Col0,Col1, ...],
                                        ...,
                                        Data from NodeN=[Col0,Col1, ...],]
    """
    num_cols = len(dtypes)

    recv_sizes = torch.empty((world_size,), dtype=torch.int64, device="cpu")
    input_splits_cpu = torch.empty((world_size,), dtype=torch.int64, device="cpu", pin_memory=True)
    input_splits_cpu.copy_(input_splits, non_blocking=True)
    torch.cuda.current_stream().synchronize()

    dist.all_to_all_single(input=input_splits_cpu, output=recv_sizes, group=group)

    prefix_sum = torch.zeros(world_size+1, dtype=torch.int64, device="cpu")
    torch.cumsum(input_splits_cpu, dim=0, out=prefix_sum[1:])

    recv_data = [[None for _ in range(num_cols)] for _ in range(world_size)]
    for c in range(num_cols):
        for i in range(world_size):
            recv_data[i][c] = torch.empty(recv_sizes[i], dtype=dtypes[c], device=target_device)

        sender_data_list    = [data[c][prefix_sum[i]:prefix_sum[i+1]] for i in range(world_size)]
        recv_list           = [recv_data[i][c] for i in range(world_size)]

        dist.all_to_all(input_tensor_list=sender_data_list, output_tensor_list=recv_list, group=group)
    return recv_data


def broadcast_ccl(world_size, data, dtypes, target_device=0, group=None):
    """
        Shuffle of data (= [Col0,Col1, ...])
        Uses collectives to distribute send/recv size information
        Returns                        [Data from Node0=[Col0,Col1, ...],
                                        ...,
                                        Data from NodeN=[Col0,Col1, ...],]
    """
    num_cols = len(dtypes)

    # unfortunately need the received tuples per rank, so no all_reduce
    send_sizes = torch.tensor([data[0].shape[0]]*world_size, dtype=torch.int64, device="cpu")
    recv_sizes = torch.empty(world_size, dtype=torch.int64, device="cpu")
    dist.all_to_all_single(input=send_sizes, output=recv_sizes, group=group)

    recv_data = [[None for _ in range(num_cols)] for _ in range(world_size)]

    for c in range(num_cols):
        for i in range(world_size):
            recv_data[i][c] = torch.empty(recv_sizes[i], dtype=dtypes[c], device=target_device)

        recv_list           = [recv_data[i][c] for i in range(world_size)]

        dist.all_gather(tensor_list=recv_list, tensor=data[c], group=group)
    return recv_data


def broadcast_ccl_combine_recv(world_size, data, dtypes, target_device=0, group=None):
    """
        Shuffle of data (= [Col0,Col1, ...])
        Uses collectives to distribute send/recv size information
        Returns                         [Col0,Col1, ...]
    """
    num_cols = len(dtypes)

    # unfortunately need the received tuples per rank, so no all_reduce
    send_sizes = torch.tensor([data[0].shape[0]]*world_size, dtype=torch.int64, device="cpu")
    recv_sizes = torch.empty(world_size, dtype=torch.int64, device="cpu")
    dist.all_to_all_single(input=send_sizes, output=recv_sizes, group=group)

    prefix_sum = torch.zeros(world_size+1, dtype=torch.int64, device="cpu")
    # print(f"{recv_sizes=}")
    torch.cumsum(recv_sizes, dim=0, out=prefix_sum[1:])
    # print(f"{prefix_sum=}")
    total_tuples = prefix_sum[-1].item()
    recv_data = [torch.empty(total_tuples, dtype=dtypes[c], device=target_device) for c in range(num_cols)]

    for c in range(num_cols):
        recv_list           = [recv_data[c][prefix_sum[i]:prefix_sum[i+1]] for i in range(world_size)]

        dist.all_gather(tensor_list=recv_list, tensor=data[c], group=group)
    return recv_data
