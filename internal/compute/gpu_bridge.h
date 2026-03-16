#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Compute TRON addresses from mnemonics on GPU.
 * Each mnemonic is stored as bytes in a flat buffer.
 *
 * mnemonic_data    - flat byte buffer containing all mnemonic strings
 * mnemonic_offsets - offset of each mnemonic in mnemonic_data
 * mnemonic_lengths - byte length of each mnemonic
 * count            - number of mnemonics
 * addresses_out    - output buffer, count * 20 bytes
 *
 * Returns 0 on success, -1 on CUDA error.
 */
int gpu_compute_addresses(
    const uint8_t *mnemonic_data,
    const int     *mnemonic_offsets,
    const int     *mnemonic_lengths,
    int            count,
    uint8_t       *addresses_out
);

/* Returns number of CUDA-capable devices (0 if none). */
int gpu_device_count(void);

/*
 * Enumerate index range on GPU: BIP39 validation + full TRON address derivation.
 *
 * known_words[12]  : BIP39 word index (0-2047) for each position, -1 = unknown
 * unknown_pos      : ordered list of unknown position indices (length = unknown_count)
 * out_addrs        : pre-allocated buffer, capacity*20 bytes
 * out_indices      : pre-allocated buffer, capacity*sizeof(int64_t) bytes
 * out_count        : set to actual number of results (valid mnemonics found)
 *
 * Returns 0 on success, -1 on CUDA error.
 */
int gpu_enumerate_compute(
    int64_t        start_idx,
    int64_t        end_idx,
    const int16_t *known_words,
    const int8_t  *unknown_pos,
    int8_t         unknown_count,
    uint8_t       *out_addrs,
    int64_t       *out_indices,
    int            capacity,
    int           *out_count
);

/*
 * Upload a bloom filter to persistent GPU memory.
 * words      : raw uint64 bitset words (BloomFilter.BitSet().Bytes())
 * word_count : number of uint64 words
 * m          : total bits (BloomFilter.Cap())
 * k          : number of hash functions (BloomFilter.K())
 * Returns 0 on success, -1 on CUDA error.
 */
int gpu_bloom_upload(const uint64_t *words, uint64_t word_count, uint64_t m, uint32_t k);

/* Release GPU bloom filter memory */
void gpu_bloom_free(void);

/* Release all persistent GPU enumerate buffers (call on shutdown) */
void gpu_enumerate_cleanup(void);

#ifdef __cplusplus
}
#endif
