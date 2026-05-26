import sys
import os
from subprocess import Popen, PIPE
from enum import Enum, auto
import re


class GPU_CON(Enum):
    # X    = auto() # Self
    # NVX  = auto() # Connection traversing a bonded set of # NVLinks
    PIX  = auto() # Connection traversing at most a single PCIe bridge
    PXB  = auto() # Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
    PHB  = auto() # Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
    NODE = auto() # Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
    SYS  = auto() # Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return NotImplemented

def execute_shell(cmd_as_array):
    process = Popen(cmd_as_array, stdout=PIPE)
    (output, err) = process.communicate()
    exit_code = process.wait()
    if exit_code != 0:
        sys.exit(f"Failed to execute: {cmd_as_array}")
    return output.decode()

def get_devices(mig_only=False):
    output = execute_shell(["nvidia-smi","-L"])
    for line in output.split("\n"):
        if not line.strip() or (mig_only and "MIG-" not in line):
            continue
        yield line.strip().split()[-1][:-1]

def get_nic_info(ibdev_name):
    # return netdev_name and ip of ib device
    # e.g. for ibdev_name="mlx5_0" might return "enp12s0np0","192.168.1.20"
    # do it naively:
    output = execute_shell(["ibdev2netdev"])
    netdev_name = None
    for line in output.split("\n"):
        line = line.strip()
        if ibdev_name in line:
            netdev_name = line.split()[-2]
            break
    else:
        print(f"Couldnt find interface for {ibdev_name=}")
    output = execute_shell(["ip", "a", "show", "dev", netdev_name])
    for line in output.split("\n"):
        line = line.strip()
        if "inet " in line:
            ip = line.split()[1][:-3]
    return netdev_name, ip

def get_hostname(args=[]):
    # use args=["-s"] for short hostname
    output = execute_shell(["hostname"]+args)
    for line in output.split("\n"):
        return line.strip()

def get_nvme_mount(nvme_name):
    # "-n" : no header
    output = execute_shell(["lsblk","/dev/"+nvme_name,"-n"])
    for line in output.split("\n"):
        return line.strip().split()[-1] + "/"


def assign_to_gpus(gpus, topo_param, target_con):
    """
        gpus: sorted as "nvidia-smi -L" / get_devices()
        topo_param = "-m" for NICs, "-nvme" for SSDs
    """
    assignment = dict()
    legend = dict()
    output = execute_shell(["nvidia-smi","topo",topo_param])

    gpu_idx = 0
    all_assigned = False
    lines = output.split("\n")
    # https://stackoverflow.com/questions/14693701/how-can-i-remove-the-ansi-escape-sequences-from-a-string-in-python
    columns = re.sub(r'(?:\x1B[@-_]|[\x80-\x9F])[0-?]*[ -/]*[@-~]', "", lines[0]).split() 


    for line in lines[1:]:
        # line = line.strip()
        if len(assignment) == len(gpus):
            all_assigned = True
        if not all_assigned and not line.startswith("GPU"):
            continue

        if all_assigned:
            # handle other lines (for legend):
            match = re.match(r"\W+((?:NIC|NVMe)\d):\W+(\w+)", line)
            if match:
                placeholder, res_name = match.groups()
                legend[placeholder] = res_name
        else:
            # handle a matrix line
            for col_idx,con in enumerate(line.split()[1:]):
                try:
                    con = GPU_CON[con] # str to enum
                except KeyError:
                    continue

                # res_placeholder = f"NIC{res_id}" if topo_param == "-m" else f"NVMe{res_id}"
                res_placeholder = columns[col_idx]
                # print(f"GPU{gpu_idx} <-> {res_placeholder}: {con}")

                if con <= target_con and res_placeholder not in assignment.values():
                    assignment[gpus[gpu_idx]] = res_placeholder
                    gpu_idx += 1
                    break

    return {gpu: legend[assigned_placeholder] for gpu, assigned_placeholder in assignment.items()}


if __name__ == '__main__':
    gpu_devices = list(get_devices())
    print(gpu_devices)
    assigned_nics = assign_to_gpus(gpu_devices,"-m", GPU_CON.PHB)
    assigned_nvmes = assign_to_gpus(gpu_devices,"-nvme", GPU_CON.PHB)
    assigned_mounts = {gpu: get_nvme_mount(nvme) for gpu,nvme in assigned_nvmes.items()}

    print(assigned_nics)
    print(assigned_nvmes)
    print(assigned_mounts)

    # check config
    # sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
    # import config
    # for gpu, res in assigned_nics.items():
    #     assert res == config.GPU_2_NIC_MAPPING[gpu][0]

    # for gpu, res in assigned_mounts.items():
    #     assert res == config.GPU_2_SSD_MAPPING[gpu]

    # print("Assignment corresponds to config")