import sys
import os
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
from subprocess import Popen
import time
import subprocess

from helpers.beemon import Beemon

if __name__ == "__main__":
    WORLD_RANK = int(os.environ['RANK'])
    # single process is enough
    if WORLD_RANK != 0:
        exit(0)

    # expects nccl and nccl-tests in home directory

    cur_env = os.environ.copy()
    cur_env["LD_LIBRARY_PATH"] = cur_env["HOME"] + "/nccl/build/lib/:" + cur_env["LD_LIBRARY_PATH"]
    cur_env["NCCL_IB_HCA"] = cur_env["LIST_OF_NCCL_IB_HCA"]
    cur_env["CUDA_VISIBLE_DEVICES"] = cur_env["LIST_OF_CUDA_VISIBLE_DEVICES"]

    variant = "alltoall_perf"
    # variant = "sendrecv_perf"
    # variant = "scatter_perf"
    repeats = int(os.environ["NUM_REPEATS"])

    beemon = Beemon()

    cmd_str = cur_env["HOME"] + f"/nccl-tests/build/{variant} -g {os.environ['WORLD_SIZE']} --minbytes={(1<<18)} --maxbytes={(1<<31)} --stepfactor=2 --local_register=1"

    print(f"NCCL_TEST_HELPER Will execute: {cmd_str=}")
    for repeat in range(repeats):
        beemon.start(WORLD_RANK, repeat)
        p = Popen(cmd_str, env=cur_env, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = p.communicate()
        exit_code = p.returncode
        print(f"Process exited with code {exit_code}")
        if stdout:
            print("STDOUT:")
            print(stdout.decode())
        if stderr:
            print("STDERR:")
            print(stderr.decode())
        beemon.stop()
