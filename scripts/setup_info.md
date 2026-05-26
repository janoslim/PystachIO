# PystachIO

# Installation

Miniconda (optional):
```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
# then
source ~/miniconda3/bin/activate
conda init --all
```

```bash
cd ~/miniconda3/envs/
mkdir distqp-env 
tar -xzf /raid/jluo/distqp-env.tar.gz -C distqp-env
conda activate distqp-env
conda-unpack # this is optional
pip3 install --pre torch==2.10.0.dev20251015+cu129 --index-url https://download.pytorch.org/whl/nightly/cu129
# then this conda env is complete

conda deactivate # if you want out of conda as usually
```

# Run
```bash
python multiproc.py run_shuffle.py 2
python multiproc.py run_tpch.py 2
python multiproc.py benchmarks/send_bench.py 2
# also check querys
export PYTHONOPTIMIZE=1 # For Benchmarking, otherwise logging will cause CUDAsync
```

## Parquet Files

TPC-H SF{100, 200, 300, ...} under: `/xssd/0/`

# NCCL
``` bash
# conda install -c nvidia cuda-toolkit
git clone https://github.com/NVIDIA/nccl
cd nccl
make -j4 src.build CUDA_HOME=/path/to/cuda NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"
# make -j32 src.build CUDA_HOME=$CONDA_PREFIX
```
# NCCL-Tests
``` bash
git clone https://github.com/NVIDIA/nccl-tests/
cd nccl-tests
make -j4 CUDA_HOME=$_CUDA_PATH NCCL_HOME="$HOME/nccl/build"
NCCL_IB_HCA="=mlx5_4,mlx5_6" LD_LIBRARY_PATH="$HOME/nccl/build/lib/:$LD_LIBRARY_PATH" NCCL_IB_DISABLE=0 NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 ./build/alltoall_perf -g 2 --minbytes=$((1<<18)) --maxbytes=$((1<<28)) --stepfactor=2
```

# Perftest with GPU support:
``` bash
git clone https://github.com/linux-rdma/perftest.git
cd perftest
# Compile outside the Conda!

# might be helpful
# _CUDA_PATH="/usr/local/cuda-13.0" CUDACXX="$_CUDA_PATH/bin/nvcc"
# nvcc_path="$_CUDA_PATH/bin/nvcc" CPATH=$_CUDA_PATH/include:$CPATH LD_LIBRARY_PATH=$_CUDA_PATH/targets/x86_64-linux/lib/:$LD_LIBRARY_PATH
# LDFLAGS="-L/usr/lib/x86_64-linux-gnu/" CFLAGS="-I/usr/include/x86_64-linux-gnu/pci/" ./configure CUDA_H_PATH=/usr/local/cuda-13.0/include/cuda.h
./autogen.sh && ./configure CUDA_H_PATH=/usr/local/cuda-13.0/include/cuda.h && make -j
# then
python multiproc.py ib_write.py 2
python multiproc.py benchmarks/perftest.py 2
```

https://github.com/linux-rdma/perftest/issues/315

# Troubleshooting:
- If RDMA communication does not work, the NICs we are trying to use are probably acting up.
Try executing
``` bash
python multiproc.py benchmarks/fix_nics.py 2
```

- NVIDIA GDS Stats might be disabled:
``` bash
# needs sudo
echo 1 > /sys/module/nvidia_fs/parameters/rw_stats_enabled
```
