#!/usr/bin/env python3
import sys
import os
from subprocess import Popen, PIPE
from datetime import datetime
import config
import helpers.topology_detect as topo

try:
    from helpers.Parquet_FIFO import shutdown_global_thread_pool
    PARQUET_FIFO_AVAILABLE = True
except ImportError:
    PARQUET_FIFO_AVAILABLE = False
    shutdown_global_thread_pool = None

##########################################################################################
cur_env = config.apply_defaults()
##########################################################################################

if len(sys.argv) < 3:
    print("Usage: python multiproc.py <program> <num_progs> [<nsys_target_file>]")
    print("<program> is started <num_progs> times with torchrun, configured visible gpus and passed target device id \"0\".")
    exit(1)

prog_name = sys.argv[1]
num_progs = int(sys.argv[2])
with_nsys = len(sys.argv) >= 4

gpu_devices = list(topo.get_devices())[:num_progs]
GPU_2_NIC_MAPPING = topo.assign_to_gpus(gpu_devices,"-m", topo.GPU_CON.PHB)
GPU_2_NVME_MAPPING = topo.assign_to_gpus(gpu_devices,"-nvme", topo.GPU_CON.PXB)
# set a default empty ssd path for all SSDs. This makes it possible to run with a single SSD,
# by using STORAGE_SPECIAL_PATH (set to e.g., "/home/pystachio/tpch")
GPU_2_SSD_MAPPING = {gpu: "" for gpu in gpu_devices}
for gpu,nvme in GPU_2_NVME_MAPPING.items():
    GPU_2_SSD_MAPPING[gpu] = topo.get_nvme_mount(nvme)

if len(GPU_2_NIC_MAPPING) != len(gpu_devices):
    sys.exit(f"No complete GPU-NIC mapping:\n{GPU_2_NIC_MAPPING}")
if len(GPU_2_SSD_MAPPING) != len(gpu_devices):
    sys.exit(f"No complete GPU-SSD mapping:\n{GPU_2_SSD_MAPPING}")

# sanity check automatic mappings against config:
if topo.get_hostname() == "dgx01.lab.dm.informatik.tu-darmstadt.de":
    print("Sanity checking config ...")
    for gpu, res in GPU_2_NIC_MAPPING.items():
        if res != config.GPU_2_NIC_MAPPING[gpu][0]:
            sys.exit(f"Topo detect NIC mapping different from config!:\n{GPU_2_NIC_MAPPING}\n{config.GPU_2_NIC_MAPPING}")
    for gpu, res in GPU_2_SSD_MAPPING.items():
        if res != config.GPU_2_SSD_MAPPING[gpu]:
            sys.exit(f"Topo detect SSD mapping different from config!:\n{GPU_2_SSD_MAPPING}\n{config.GPU_2_SSD_MAPPING}")

if cur_env.get("MULTI_SSD_READER", "False") != "False": # could also be numbers :)
    if len(gpu_devices) > 1:
        sys.exit("ERROR: MULTI_SSD_READER=True only makes sense for single GPU setups!")

# for ib_write_bw: set IB_MASTER_IP to ip of first device and NIC
_,IB_MASTER_IP = topo.get_nic_info(GPU_2_NIC_MAPPING[gpu_devices[0]])
cur_env["IB_MASTER_IP"] = IB_MASTER_IP
# for nccl-tests: NCCL_IB_HCA is set to one GPU/NIC, put all GPU/NICs into custom var
cur_env["LIST_OF_CUDA_VISIBLE_DEVICES"] = ",".join([gpu_id for gpu_id in gpu_devices])
cur_env["LIST_OF_NCCL_IB_HCA"] = "=" + ",".join([GPU_2_NIC_MAPPING[gpu_id] for gpu_id in gpu_devices])

total_nodes = int(cur_env.get("TOTAL_NODES",num_progs))
node_offset = int(cur_env.get("NODE_OFFSET",0))

procs = []
outfiles = []
for p in range(num_progs):
    outfiles.append(open(cur_env["STDOUT_TEMPLATE"].format(p), "a"))
    outfiles[-1].write(f"####### started: {str(datetime.now())}\n")
    outfiles[-1].flush()
for p in range(num_progs):
    cur_env["CUDA_VISIBLE_DEVICES"] = gpu_devices[p]
    ibdev_name = GPU_2_NIC_MAPPING[gpu_devices[p]]
    netdev_name,_ = topo.get_nic_info(ibdev_name)
    cur_env["NCCL_IB_HCA"] = "="+ibdev_name
    cur_env["NCCL_SOCKET_IFNAME"] = "="+netdev_name
    cur_env["STORAGE_BASE_PATH"] = GPU_2_SSD_MAPPING[gpu_devices[p]] + cur_env["STORAGE_SPECIAL_PATH"]
    # cur_env["GLOO_SOCKET_IFNAME"] = "="+netdev_name

    # Simple logging for sanity check once NIC failes, see issue: https://github.com/nboeschen/pytorchio/issues/18
    outfiles[p].write(f"####### GPU: {cur_env['CUDA_VISIBLE_DEVICES']}, SSD: {GPU_2_SSD_MAPPING[gpu_devices[p]]}, IB_DEV: {ibdev_name}, NET_DEV: {netdev_name}, NCCL_IB_HCA: {cur_env['NCCL_IB_HCA']}, NCCL_SOCKET_IFNAME: {cur_env['NCCL_SOCKET_IFNAME']}\n")
    outfiles[p].flush()

    # if p == 0:
    #     cmd_str = f"gdb -ex='set disable-randomization off' -ex=run --batch --args python -m torch.distributed.run --nproc_per_node=1 --nnodes={num_progs} --node_rank={p} --master_addr={cur_env['MASTER_ADDR']} --master_port={cur_env['MASTER_PORT']} {prog_name} 0"
    # else:
    cmd_str = f"python -m torch.distributed.run --nproc_per_node=1 --nnodes={total_nodes} --node_rank={p+node_offset} --master_addr={cur_env['MASTER_ADDR']} --master_port={cur_env['MASTER_PORT']} {prog_name} 0"

    if with_nsys and p == 0:
        # cmd_str = f"nsys profile --nic-metrics=true -o {sys.argv[3]} -f true " + cmd_str
        # cmd_str = f"nsys profile --trace=cuda,nvtx,osrt --gpu-metrics-devices=0 -o {sys.argv[3]} -f true {cmd_str}"
        cmd_str = f"nsys profile --trace=cuda,nvtx,osrt -o {sys.argv[3]} -f true {cmd_str}"
        # cmd_str = f"nsys profile --nic-metrics=true --gpu-metrics-device=all --show-output=false -o {sys.argv[3]} -f true " + cmd_str
    if cur_env["USE_OWN_STDOUT"] == "True":
        procs.append(Popen(cmd_str, stdout=outfiles[p], stderr=outfiles[p], env=cur_env, shell=True))
    else:
        procs.append(Popen(cmd_str, env=cur_env, shell=True))

if PARQUET_FIFO_AVAILABLE:
    shutdown_global_thread_pool() # In storage FIFO

for p in procs:
    p.wait()

for f in outfiles:
    f.close()
