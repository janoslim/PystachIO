import sys
import os
from subprocess import Popen, PIPE
import time

NICS = ["mlx5_0","mlx5_1","mlx5_2","mlx5_3","mlx5_4","mlx5_5","mlx5_6","mlx5_7"]

def execute_shell(cmd_as_array):
    process = Popen(cmd_as_array, stdout=PIPE)
    (output, err) = process.communicate()
    exit_code = process.wait()
    if exit_code != 0:
        print(f"##### Error:",err)
    return exit_code, output

def get_nic_info(ibdev_name):
    # return netdev_name and ip of ib device
    # e.g. for ibdev_name="mlx5_0" might return "enp12s0np0","192.168.1.20"
    # do it naively:
    _, output = execute_shell(["ibdev2netdev"])
    netdev_name = None
    for line in output.decode().split("\n"):
        line = line.strip()
        if ibdev_name in line:
            netdev_name = line.split()[-2]
            break
    else:
        print(f"Couldnt find interface for {ibdev_name=}")
    _, output = execute_shell(["ip", "a", "show", "dev", netdev_name])
    for line in output.decode().split("\n"):
        line = line.strip()
        if "inet " in line:
            ip = line.split()[1][:-3]
    return netdev_name, ip


def small_perftest(cmd_str_without_nic, rank, nic, master_ip):
    cmd_str = cmd_str_without_nic + f" {nic} "
    if rank != 0:
        cmd_str += f" {master_ip}"
        time.sleep(1)
    print(f"#### PERFTEST_HELPER Will execute: {cmd_str=}")
    p = Popen(cmd_str, shell=True)
    exit_code = p.wait()
    return exit_code


if __name__ == "__main__":
    rank = int(os.environ['RANK'])
    if rank >= 2:
        print(f"#### PERFTEST_HELPER will not run for RANK={rank}")
        sys.exit(0)

    use_cuda = os.environ.setdefault("PERFTEST_USE_CUDA","True") == "True"
    variant = os.environ.setdefault("PERFTEST_VARIANT","ib_write_bw")

    # use the nccl_ib_hca and the torchrun vars
    # print(f"{os.environ['WORLD_SIZE']=}, {os.environ['RANK']=}, {os.environ['IB_MASTER_IP']=}")
    cmd_str_without_nic = f"$HOME/perftest/{variant} -F {'--use_cuda=0' if use_cuda else ''} -d"

    for outer_nic in NICS:
        for inner_nic in NICS:
            if inner_nic == outer_nic:
                continue

            cur_nics = [outer_nic,inner_nic]
            print(f"Would execute with {cur_nics=}")
            _, master_ip = get_nic_info(outer_nic)
            exit_code = small_perftest(cmd_str_without_nic, rank, cur_nics[rank], master_ip)
            print("#### exit code", exit_code)


