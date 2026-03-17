/*
 * gpu_batch.cu – Batch TRON address derivation pipeline on CUDA
 *
 * Reuses the shared device helpers from gpu_enumerate.cu while keeping the
 * batch kernel in its own translation unit, so batch/enumerate register usage
 * remains isolated and multi-GPU host ABI stays unchanged.
 */

#define GPU_ENUMERATE_SHARED_ONLY
#undef __device__
#define __device__ static __attribute__((device))
#include "gpu_enumerate.cu"
#undef __device__

/* ================================================================
 * Kernel: one thread per mnemonic
 * ================================================================ */
__global__ void tron_batch_kernel(
    const uint8_t *mdata,
    const int     *moff,
    const int     *mlen,
    int            count,
    uint8_t       *out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    const uint8_t *mn = mdata + moff[idx];
    uint32_t       ml = (uint32_t)mlen[idx];

    /* Match enumerate's scratch reuse so both entry points share the same
     * crypto implementation and a very similar stack profile. */
    uint8_t tmp[64];
    uint8_t pub[64];
    en_pbkdf2_hmac_sha512(mn, ml, tmp);
    en_bip44_tron(tmp, tmp);
    en_priv_to_upub64(tmp, pub);
    en_keccak256(pub, 64, tmp);

    memcpy(out + (int64_t)idx * 20, tmp + 12, 20);
}

__global__ void pbkdf2_batch_kernel(
    const uint8_t *mdata,
    const int     *moff,
    const int     *mlen,
    int            count,
    uint8_t       *out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    const uint8_t *mn = mdata + moff[idx];
    uint32_t       ml = (uint32_t)mlen[idx];

    en_pbkdf2_hmac_sha512(mn, ml, out + (int64_t)idx * 64);
}

/* ================================================================
 * Host-side persistent state – one slot per GPU device.
 * Reuses batch buffers across Compute() calls to avoid repeated
 * cudaMalloc/cudaFree overhead on the hot path.
 * ================================================================ */
#define MAX_DEVICES 8

typedef struct {
    uint8_t *d_data;
    int     *d_off;
    int     *d_len;
    uint8_t *d_out;
    uint8_t *d_seed;
    int      data_capacity;
    int      count_capacity;
    int      stack_limit_set;
} BatchState;

static BatchState g_batch[MAX_DEVICES];

static void free_count_buffers(BatchState *bs)
{
    if (bs->d_off) { cudaFree(bs->d_off); bs->d_off = NULL; }
    if (bs->d_len) { cudaFree(bs->d_len); bs->d_len = NULL; }
    if (bs->d_out) { cudaFree(bs->d_out); bs->d_out = NULL; }
    if (bs->d_seed) { cudaFree(bs->d_seed); bs->d_seed = NULL; }
    bs->count_capacity = 0;
}

static int ensure_batch_buffers(int device_id, int data_bytes, int count)
{
    BatchState *bs = &g_batch[device_id];

    if (data_bytes > bs->data_capacity) {
        if (bs->d_data) { cudaFree(bs->d_data); bs->d_data = NULL; }
        bs->data_capacity = 0;
        if (cudaMalloc(&bs->d_data, (size_t)data_bytes) != cudaSuccess) return -1;
        bs->data_capacity = data_bytes;
    }

    if (count > bs->count_capacity) {
        free_count_buffers(bs);
        if (cudaMalloc(&bs->d_off, (size_t)count * sizeof(int)) != cudaSuccess) goto fail;
        if (cudaMalloc(&bs->d_len, (size_t)count * sizeof(int)) != cudaSuccess) goto fail;
        if (cudaMalloc(&bs->d_out, (size_t)count * 20)          != cudaSuccess) goto fail;
        if (cudaMalloc(&bs->d_seed, (size_t)count * 64)         != cudaSuccess) goto fail;
        bs->count_capacity = count;
    }

    if (!bs->stack_limit_set) {
        if (cudaDeviceSetLimit(cudaLimitStackSize, 65536) != cudaSuccess) return -1;
        bs->stack_limit_set = 1;
    }
    return 0;

fail:
    free_count_buffers(bs);
    return -1;
}

extern "C" void gpu_batch_cleanup(int device_id)
{
    if (device_id < 0 || device_id >= MAX_DEVICES) return;
    if (cudaSetDevice(device_id) != cudaSuccess) return;

    BatchState *bs = &g_batch[device_id];
    if (bs->d_data) { cudaFree(bs->d_data); bs->d_data = NULL; }
    bs->data_capacity = 0;
    free_count_buffers(bs);
}

/* ================================================================
 * Host functions
 * ================================================================ */
extern "C" int gpu_compute_addresses(
    int            device_id,
    const uint8_t *mnemonic_data,
    const int     *mnemonic_offsets,
    const int     *mnemonic_lengths,
    int            count,
    uint8_t       *addresses_out)
{
    if (device_id < 0 || device_id >= MAX_DEVICES) return -1;
    if (cudaSetDevice(device_id) != cudaSuccess) return -1;
    if (count <= 0) return 0;

    int max_data = 0;
    for (int i = 0; i < count; i++) {
        int end = mnemonic_offsets[i] + mnemonic_lengths[i];
        if (end > max_data) max_data = end;
    }
    if (max_data <= 0) max_data = 1;

    if (ensure_batch_buffers(device_id, max_data, count) != 0) return -1;
    BatchState *state = &g_batch[device_id];

    if (cudaMemcpy(state->d_data, mnemonic_data, (size_t)max_data, cudaMemcpyHostToDevice) != cudaSuccess) return -1;
    if (cudaMemcpy(state->d_off, mnemonic_offsets, (size_t)count * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) return -1;
    if (cudaMemcpy(state->d_len, mnemonic_lengths, (size_t)count * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) return -1;

    {
        int block_size = 256;
        int num_blocks = (count + block_size - 1) / block_size;
        tron_batch_kernel<<<num_blocks, block_size>>>(
            state->d_data, state->d_off, state->d_len, count, state->d_out);
        if (cudaDeviceSynchronize() != cudaSuccess) return -1;
    }

    if (cudaMemcpy(addresses_out, state->d_out, (size_t)count * 20, cudaMemcpyDeviceToHost) != cudaSuccess) return -1;
    return 0;
}

extern "C" int gpu_compute_pbkdf2_seeds(
    int            device_id,
    const uint8_t *mnemonic_data,
    const int     *mnemonic_offsets,
    const int     *mnemonic_lengths,
    int            count,
    uint8_t       *seeds_out)
{
    if (device_id < 0 || device_id >= MAX_DEVICES) return -1;
    if (cudaSetDevice(device_id) != cudaSuccess) return -1;
    if (count <= 0) return 0;

    int max_data = 0;
    for (int i = 0; i < count; i++) {
        int end = mnemonic_offsets[i] + mnemonic_lengths[i];
        if (end > max_data) max_data = end;
    }
    if (max_data <= 0) max_data = 1;

    if (ensure_batch_buffers(device_id, max_data, count) != 0) return -1;
    BatchState *state = &g_batch[device_id];

    if (cudaMemcpy(state->d_data, mnemonic_data, (size_t)max_data, cudaMemcpyHostToDevice) != cudaSuccess) return -1;
    if (cudaMemcpy(state->d_off, mnemonic_offsets, (size_t)count * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) return -1;
    if (cudaMemcpy(state->d_len, mnemonic_lengths, (size_t)count * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) return -1;

    {
        int block_size = 256;
        int num_blocks = (count + block_size - 1) / block_size;
        pbkdf2_batch_kernel<<<num_blocks, block_size>>>(
            state->d_data, state->d_off, state->d_len, count, state->d_seed);
        if (cudaDeviceSynchronize() != cudaSuccess) return -1;
    }

    if (cudaMemcpy(seeds_out, state->d_seed, (size_t)count * 64, cudaMemcpyDeviceToHost) != cudaSuccess) return -1;
    return 0;
}
