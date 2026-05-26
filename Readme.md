# PystachIO

This is the source code for our published paper at VLDB'26: PystachIO: Efficient Distributed GPU Query Processing with PyTorch over Fast Networks & Fast Storage.
Jigao Luo and Nils Boeschen are the main developers of the code and contributed equally.

## Abstract

The AI hardware boom has led modern data centers to adopt HPC-style architectures centered on distributed, GPU-centric computation. 
Large GPU clusters interconnected by fast RDMA networks and backed by high-bandwidth NVMe storage enable scalable computation and rapid access to storage-resident data. 
Tensor computation runtimes (TCRs), such as PyTorch, originally designed for AI workloads, have recently been shown to accelerate analytical workloads. 
However, prior work has primarily considered settings where the data fits in aggregated GPU memory. 
In this paper, we systematically study how TCRs can support scalable, distributed query processing for large-scale, storage-resident OLAP workloads.
Although TCRs provide abstractions for network and storage I/O, naive use often underutilizes GPU and I/O bandwidth due to insufficient overlap between computation and data movement. 
As a core contribution, we present PystachIO, a prototype of a PyTorch-based distributed OLAP engine that combines fast network and storage I/O with key optimizations to maximize GPU, network, and storage utilization. 
Our evaluation shows up to 3× end-to-end speedups over existing distributed GPU-based query processing approaches.

## Setup

All experiments are run on an NVIDIA DGX system with Ubuntu 22.04 (DGX OS 6), Linux kernel 5.15, CUDA 12.9, and PyTorch 2.10. 
The system has two AMD EPYC 7742 64-core CPUs, 8 NVIDIA A100 GPUs with 40 GiB memory each, 8 Mellanox ConnectX-6 200G InfiniBand/Ethernet NICs, and 8 NVMe SSDs with up to 6.4 GiB/s read bandwidth. 
Each GPU is paired with its nearest NIC and SSD for network and storage I/O.

## Code Repository for PystachIO

Overview:
- `modules/`: Main sources for distributed and table scan operators over fast networks and fast storage.
- `tpch/`: Implementation of distributed TPCH queries over Parquet files.
- `scripts/`: Utilities.
- `benchmarks/`: Microbenchmarks, performance tests and related utilities for distributed or parquet processing.

The setup is based on a conda environment with customized cuDF code, executed on a DGX A100 machine.
Most configuration options can be found in `config.py`.
`multiproc.py` is a starting script that prefers specific environment variable for started processes (one per GPU),
Examples:
```
# Run benchmark of shuffle of two tables with 4 columns each, on 6 nodes (GPUs)
SHUFFLE_LEFT_TUPLES=120000000 SHUFFLE_RIGHT_TUPLES=360000000 NUM_COLS=4 python multiproc.py run_shuffle.py 6

# Run tpch query 3 on scale factor 1000, using 4 nodes
TPCH_SCALE_FACTOR=1000 TPCH_QUERY=4 python multiproc.py run_tpch.py 4
```

Run scripts for experiments can be found in `scripts/` folder.

---

## Build

## Download and install custom cudf
Miniconda (optional):
```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate
conda init --all
```
Create env:
```bash
git clone --depth 1 git@github.com:JigaoLuo/MS-repro-cuDF.git # https://github.com/JigaoLuo/MS-repro-cuDF
cd MS-repro-cuDF/
conda env create --name pystachio-env --file conda/environments/all_cuda-129_arch-x86_64.yaml
conda activate pystachio-env
conda install -c conda-forge humanize
./build.sh --ptds libcudf pylibcudf # more CPUs will help to compile
pip3 install torch --index-url https://download.pytorch.org/whl/cu129
```

## Configure:
Network:
```bash
# check if topology is correctly detected:
python helpers/topology_detect.py
# set MASTER_ADDR to an IP of the master node
ip a
export MASTER_ADDR="<master IP>"
```

Parquet file storage:
- use `STORAGE_SPECIAL_PATH`, since detected `GPU_2_NVME_MAPPING` will be empty, and `GPU_2_SSD_MAPPING` will be `""`.
    - Final folder path = (`GPU_2_SSD_MAPPING` + `STORAGE_SPECIAL_PATH`) + `"/tpch-sf{SCALE_FACTOR}/""`
    - e.g. if parquets are in `"/home/pystachio/tpch-sf1000/"`, set `STORAGE_SPECIAL_PATH` to `"/home/pystachio/"`
- also enable `KVIKIO_COMPAT_MODE` for non-GDS access
```bash
export STORAGE_SPECIAL_PATH="/home/pystachio/"
export KVIKIO_COMPAT_MODE=ON
```

## Datasets: Parquet files

TLDR: 
1. Generate TPC-H from `DuckDB`
2. Use our rewriter to output GPU-optimized Parquet files: https://github.com/DataManagementLab/ParquetRewriter


## Run:
```bash
# Set STORAGE_SPECIAL_PATH to the common prefix, e.g., /home/users/tpch, if you have directories such as tpch-100/ and tpch-200/
# Example: export STORAGE_SPECIAL_PATH="/home/users/tpch"
bash tpch_sf_scaling_8GPUs.sh
```
