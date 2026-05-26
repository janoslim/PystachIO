import cupy
import torch
import kvikio
import kvikio.nvcomp
from kvikio._lib.arr import asarray

class CompressHelper:
    # This essentially mimics kvikios Manager (high-level) interface
    def __init__(self, device, dtype, max_decomp_tuples):
        self.max_decomp_tuples = max_decomp_tuples
        self.dtype = dtype
        self.device = device
        self.lib_manager = kvikio._lib.libnvcomp._GdeflateManager(65536,0,0,device)


    def compress(self, tensor):
        comp_config = self.lib_manager.configure_compression(tensor.nelement()*tensor.element_size())
        compressed = torch.empty((comp_config["max_compressed_buffer_size"],), dtype=torch.int8, device=self.device)
        cupy_data = cupy.from_dlpack(tensor)

        # TODO: not sure this is really necessary
        compressed_bytes = self.lib_manager.compress(asarray(cupy_data),asarray(compressed))

        return compressed[:compressed_bytes]

    def decompress(self, compressed):
        decomp_config = self.lib_manager.configure_decompression_with_compressed_buffer(asarray(compressed))
        decompressed = torch.empty((decomp_config["decomp_data_size"],), dtype=torch.int8, device=self.device)

        self.lib_manager.decompress(asarray(decompressed),asarray(compressed))
        return decompressed.view(self.dtype)