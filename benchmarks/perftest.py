import sys
import os
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
from subprocess import Popen
import time

from helpers.beemon import Beemon

if __name__ == "__main__":
    WORLD_RANK = int(os.environ['RANK'])
    if WORLD_RANK >= 2:
        print(f"#### PERFTEST_HELPER will not run for RANK={os.environ['RANK']}")
        sys.exit(0)

    use_cuda = os.environ.setdefault("PERFTEST_USE_CUDA","True") == "True"
    variant = os.environ.setdefault("PERFTEST_VARIANT","ib_write_bw")

    beemon = Beemon()
    beemon.start(WORLD_RANK, 0)

    # use the nccl_ib_hca and the torchrun vars
    # print(f"{os.environ['WORLD_SIZE']=}, {os.environ['RANK']=}, {os.environ['IB_MASTER_IP']=}")
    # cmd_str = f"$HOME/perftest/{variant} -F -c UC -d {os.environ['NCCL_IB_HCA'].removeprefix('=')} -a" 
    cmd_str = f"$HOME/perftest/{variant} -F -d {os.environ['NCCL_IB_HCA'].removeprefix('=')} -a"
    if use_cuda:
        cmd_str += " --use_cuda=0"
    if int(os.environ['RANK']) != 0:
        cmd_str += f" {os.environ['IB_MASTER_IP']}"
        time.sleep(1)
    print(f"PERFTEST_HELPER Will execute: {cmd_str=}")
    p = Popen(cmd_str, shell=True)
    exit_code = p.wait()

    beemon.stop()
