import os
from datetime import datetime
from subprocess import Popen, PIPE
import signal

# Requirement:
# echo 1 > /sys/module/nvidia_fs/parameters/rw_stats_enabled
# cat /proc/driver/nvidia-fs/stats 

DATETIME = datetime.now().strftime('%Y-%m-%d.%H-%M-%S') # static for all repeats

class Beemon:
    def __init__(self):
        self.proc = None
        self.file = None
        self.is_enabled = os.environ.get("USE_BMON", "False") == "True"

    def start(self, world_rank: int = 0, num_repeat: int = 0):
        self.num_repeat = num_repeat
        self.with_header = self.num_repeat == 0

        if world_rank != 0 or not self.is_enabled:
            return False
        try:
            os.makedirs("results/beemon", exist_ok=True)

            # this splits LIST_OF_NCCL_IB_HCA "=mlx5_0,mlx5_1,mlx5_2" into ['0,', '1,', '2'], then joins as "0,1,2"
            nic_numbers = os.environ["LIST_OF_NCCL_IB_HCA"][1:].replace("mlx5_","")

            filename = os.environ.get("BMON_FILE",f"results/beemon/beemon_{DATETIME}.csv")

            self.file = open(filename, "a")
            bmon_path = os.path.join(os.path.expanduser("~"), "beemon", "beemon") # I assume the binary in the path: ~/beemon/beemon
            bmon_time = os.environ.get("BMON_TIME", "50")
            bmon_list = [bmon_path, bmon_time, f"--nics={nic_numbers}"] # watch all relevant nics
            if not self.with_header:
                bmon_list.append("--with-header=false")
            bmon_list.append(f"--header-suffix=repeat")
            bmon_list.append(f"--row-suffix={self.num_repeat}")
            bmon_list.append(f"--gpu={os.environ['CUDA_VISIBLE_DEVICES']}")
            self.proc = Popen(bmon_list, stdout=self.file, shell=False)
            return True
        except Exception as e:
            print(f"Warning: Failed to start beemon monitor: {e}")
            self.cleanup()
            return False
    
    def stop(self):
        if not self.is_enabled or self.proc is None:
            return
        try:
            self.proc.send_signal(signal.SIGINT)
            self.proc.wait()
        except Exception as e:
            print(f"Warning: Error stopping beemon monitor process: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        if self.file:
            try:
                self.file.close()
            except Exception:
                pass
            self.file = None
        self.proc = None
