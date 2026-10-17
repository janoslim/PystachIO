"""Probe-side key filter of the spill mode's Grace join (SHUFFLE_TYPE=GRACE, patch 0003).

Built once the join has chosen k, from the build keys of the partitions it spills (k..P-1); a join that keeps every
partition resident (k = P) builds none. Probe rows of a spilled partition whose key the filter rejects cannot join:
the probe side's partition step drops them before they are written (partition_ids: one kernel computes every row's
partition id and tests only the rows of spilled partitions; rows of resident partitions are probed at once anyway).
Two kinds:
* exact bitmap: one bit per key value from the smallest to the largest build key, used when it is no larger than
  the Bloom filter would be (dense key domains such as partkey);
* register-blocked Bloom filter: BLOOM_BITS_PER_KEY bits per build row in 64-bit words; a key sets BLOOM_HASHES
  bits inside one word, all taken from one 64-bit hash, so testing it costs one memory access. 12 bits and 5 bits
  per key give about 1 % false positives (Poisson model of the keys per word); each filter also measures its own
  rate (false_positive_rate).
Both bit arrays are packed and filled with atomicOr, so insert and test are small CUDA kernels (cupy.RawModule,
compiled by NVRTC on first use; cupy is in the PystachIO env) on the caller's current CUDA stream.
"""
import cupy
import numpy as np
import torch

BLOOM_BITS_PER_KEY = 12
BLOOM_HASHES = 5             # bit positions come from 6-bit fields of the hash's low 32 bits
NON_MEMBER_PROBES = 1 << 20  # keys above the largest build key, tested to measure the false-positive rate
KEYS_PER_STEP = 1 << 26      # build keys concatenated per kernel launch (512 MiB of int64)
THREADS_PER_BLOCK = 256

KERNEL_SOURCE = r"""
__device__ unsigned long long mix64(unsigned long long x) {  // splitmix64 finalizer
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27; x *= 0x94d049bb133111ebULL;
    x ^= x >> 31; return x;
}

// A key's word: the hash's high 32 bits reduced to [0, word_count). Its bits in that word: the low 32 bits.
__device__ unsigned long long bloom_word(unsigned long long hash, unsigned long long word_count) {
    return ((hash >> 32) * word_count) >> 32;
}

// BLOOM_HASHES is a compile-time constant (NVRTC option -D), so the loop unrolls.
__device__ unsigned long long bloom_mask(unsigned long long hash) {
    unsigned long long mask = 0;
#pragma unroll
    for (int i = 0; i < BLOOM_HASHES; ++i) {
        mask |= 1ULL << ((hash >> (6 * i)) & 63);
    }
    return mask;
}

__device__ bool bloom_contains(const unsigned long long* words, unsigned long long word_count, long long key) {
    unsigned long long hash = mix64((unsigned long long)key);
    unsigned long long mask = bloom_mask(hash);
    return (words[bloom_word(hash, word_count)] & mask) == mask;
}

__device__ bool bitmap_contains(const unsigned long long* words, long long min_key, long long bit_count,
                                long long key) {
    long long bit = key - min_key;
    return bit >= 0 && bit < bit_count && ((words[bit >> 6] >> (bit & 63)) & 1ULL);
}

// key mod partitions in [0, partitions), like torch.remainder.
__device__ long long partition_of(long long key, long long partitions) {
    long long partition = key % partitions;
    return partition < 0 ? partition + partitions : partition;
}

extern "C" __global__ void bloom_insert(unsigned long long* words, unsigned long long word_count,
                                        const long long* keys, long long count) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= count) return;
    unsigned long long hash = mix64((unsigned long long)keys[i]);
    atomicOr(&words[bloom_word(hash, word_count)], bloom_mask(hash));
}

extern "C" __global__ void bloom_test(const unsigned long long* words, unsigned long long word_count,
                                      const long long* keys, long long count, bool* found) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= count) return;
    found[i] = bloom_contains(words, word_count, keys[i]);
}

extern "C" __global__ void bitmap_insert(unsigned long long* words, long long min_key, const long long* keys,
                                         long long count) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= count) return;
    long long bit = keys[i] - min_key;
    atomicOr(&words[bit >> 6], 1ULL << (bit & 63));
}

extern "C" __global__ void bitmap_test(const unsigned long long* words, long long min_key, long long bit_count,
                                       const long long* keys, long long count, bool* found) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= count) return;
    found[i] = bitmap_contains(words, min_key, bit_count, keys[i]);
}

// The uint8 partition id of every key; a key of a spilled partition (id >= resident) that the filter rejects gets the
// id `partitions` instead, which the partition step does not gather.
extern "C" __global__ void bloom_partition_ids(const unsigned long long* words, unsigned long long word_count,
                                               const long long* keys, long long count, long long partitions,
                                               long long resident, unsigned char* ids) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= count) return;
    long long key = keys[i];
    long long partition = partition_of(key, partitions);
    if (partition >= resident && !bloom_contains(words, word_count, key)) partition = partitions;
    ids[i] = (unsigned char)partition;
}

extern "C" __global__ void bitmap_partition_ids(const unsigned long long* words, long long min_key, long long bit_count,
                                                const long long* keys, long long count, long long partitions,
                                                long long resident, unsigned char* ids) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= count) return;
    long long key = keys[i];
    long long partition = partition_of(key, partitions);
    if (partition >= resident && !bitmap_contains(words, min_key, bit_count, key)) partition = partitions;
    ids[i] = (unsigned char)partition;
}
"""
KERNEL_NAMES = ["bloom_insert", "bloom_test", "bitmap_insert", "bitmap_test", "bloom_partition_ids",
                "bitmap_partition_ids"]

_kernels = None  # the compiled cupy.RawModule, built once per process


def kernels() -> cupy.RawModule:
    """The filter's CUDA kernels, compiled on first use (GraceHashJoin's constructor, before the query runs)."""
    global _kernels
    if _kernels is None:
        _kernels = cupy.RawModule(code=KERNEL_SOURCE, options=(f"-DBLOOM_HASHES={BLOOM_HASHES}",))
        for name in KERNEL_NAMES:
            _kernels.get_function(name)
    return _kernels


def launch(name: str, count: int, *args) -> None:
    """Run one kernel over `count` keys on torch's current CUDA stream."""
    blocks = (count + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    with cupy.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
        kernels().get_function(name)((blocks,), (THREADS_PER_BLOCK,), args)


def filter_bytes_at_most(build_keys: int) -> int:
    """GPU bytes of the bits of a filter over build_keys keys, at most: the Bloom filter's (the exact bitmap is chosen
    only when it is not larger). GraceHashJoin counts them when it chooses k."""
    return (build_keys * BLOOM_BITS_PER_KEY + 63) // 64 * 8


def key_steps(key_chunks: list[torch.Tensor]):
    """The build keys as int64 tensors of about KEYS_PER_STEP keys each (few kernel launches for many chunks)."""
    group = []
    group_keys = 0
    for chunk in key_chunks:
        group.append(chunk)
        group_keys += chunk.shape[0]
        if group_keys >= KEYS_PER_STEP:
            yield torch.cat(group).to(torch.int64)
            group = []
            group_keys = 0
    if group:
        yield torch.cat(group).to(torch.int64)


class KeyFilter:
    """A filter over the build keys of a join's spilled partitions: exact bitmap or Bloom filter (module docstring)."""

    def __init__(self, key_chunks: list[torch.Tensor]):
        """key_chunks: the build keys, in chunks (any integer dtype)."""
        key_chunks = [chunk for chunk in key_chunks if chunk.shape[0] > 0]
        self.build_keys = sum(chunk.shape[0] for chunk in key_chunks)
        # Without build keys: a bitmap of no bits, which rejects every key (no probe row can join). The one word is
        # never read; it only gives the kernels a valid pointer.
        self.kind = "bitmap"
        self.min_key = 0
        self.bit_count = 0
        self.word_count = 0
        self.size_bytes = 0
        self.words = torch.zeros(1, dtype=torch.int64, device="cuda")
        self.false_positive_rate = 0.0
        if self.build_keys > 0:
            self.build(key_chunks)

    def build(self, key_chunks: list[torch.Tensor]) -> None:
        """Choose the kind from the key range, allocate the packed bits and insert every key."""
        bounds = torch.stack([torch.stack(torch.aminmax(keys)) for keys in key_steps(key_chunks)])
        self.min_key = int(bounds[:, 0].min().item())
        self.max_key = int(bounds[:, 1].max().item())
        domain_bits = self.max_key - self.min_key + 1
        bloom_bits = self.build_keys * BLOOM_BITS_PER_KEY
        # The exact bitmap wins whenever it is not larger than the Bloom filter (dense keys).
        self.kind = "bitmap" if domain_bits <= bloom_bits else "bloom"
        self.bit_count = domain_bits if self.kind == "bitmap" else bloom_bits
        self.word_count = (self.bit_count + 63) // 64
        self.size_bytes = self.word_count * 8
        self.words = torch.zeros(self.word_count, dtype=torch.int64, device="cuda")
        for keys in key_steps(key_chunks):
            self.insert(keys)
        if self.kind == "bloom":
            # Keys above the largest build key are not build keys: the share the filter accepts is its FP rate.
            non_members = torch.arange(self.max_key + 1, self.max_key + 1 + NON_MEMBER_PROBES, device="cuda")
            self.false_positive_rate = self.might_contain(non_members).sum().item() / NON_MEMBER_PROBES

    def insert(self, keys: torch.Tensor) -> None:
        """Set the bits of int64 keys."""
        count = keys.shape[0]
        if self.kind == "bitmap":
            launch("bitmap_insert", count, cupy.asarray(self.words), np.int64(self.min_key), cupy.asarray(keys),
                   np.int64(count))
        else:
            launch("bloom_insert", count, cupy.asarray(self.words), np.uint64(self.word_count), cupy.asarray(keys),
                   np.int64(count))

    def might_contain(self, keys: torch.Tensor) -> torch.Tensor:
        """Bool mask over keys: False = certainly not a build key; True = a build key (or a Bloom false positive)."""
        count = keys.shape[0]
        found = torch.zeros(count, dtype=torch.bool, device="cuda")
        if count == 0:
            return found
        keys = keys.to(torch.int64).contiguous()
        if self.kind == "bitmap":
            launch("bitmap_test", count, cupy.asarray(self.words), np.int64(self.min_key), np.int64(self.bit_count),
                   cupy.asarray(keys), np.int64(count), cupy.asarray(found))
        else:
            launch("bloom_test", count, cupy.asarray(self.words), np.uint64(self.word_count), cupy.asarray(keys),
                   np.int64(count), cupy.asarray(found))
        return found

    def partition_ids(self, keys: torch.Tensor, partitions: int, resident: int) -> torch.Tensor:
        """uint8 partition id of every key (key % partitions, like torch.remainder); keys of the spilled partitions
        (ids resident..partitions-1) that the filter rejects get the id `partitions`: their rows cannot join."""
        count = keys.shape[0]
        ids = torch.empty(count, dtype=torch.uint8, device="cuda")
        if count == 0:
            return ids
        keys = keys.to(torch.int64).contiguous()
        if self.kind == "bitmap":
            launch("bitmap_partition_ids", count, cupy.asarray(self.words), np.int64(self.min_key),
                   np.int64(self.bit_count), cupy.asarray(keys), np.int64(count), np.int64(partitions),
                   np.int64(resident), cupy.asarray(ids))
        else:
            launch("bloom_partition_ids", count, cupy.asarray(self.words), np.uint64(self.word_count),
                   cupy.asarray(keys), np.int64(count), np.int64(partitions), np.int64(resident), cupy.asarray(ids))
        return ids
