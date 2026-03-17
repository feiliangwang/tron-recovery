#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Returns number of CUDA-capable devices (0 if none). */
int gpu_device_count(void);

/*
 * Compute TRON addresses from mnemonics on the specified GPU device.
 * Each mnemonic is stored as bytes in a flat buffer.
 *
 * device_id        - CUDA device index (0-based)
 * mnemonic_data    - flat byte buffer containing all mnemonic strings
 * mnemonic_offsets - offset of each mnemonic in mnemonic_data
 * mnemonic_lengths - byte length of each mnemonic
 * count            - number of mnemonics
 * addresses_out    - output buffer, count * 20 bytes
 *
 * Returns 0 on success, -1 on CUDA error.
 */
int gpu_compute_addresses(
    int            device_id,
    const uint8_t *mnemonic_data,
    const int     *mnemonic_offsets,
    const int     *mnemonic_lengths,
    int            count,
    uint8_t       *addresses_out
);

/*
 * Enumerate index range on the specified GPU device:
 * BIP39 validation + full TRON address derivation.
 *
 * device_id    - CUDA device index (0-based)
 * known_words[12]  : BIP39 word index (0-2047) for each position, -1 = unknown
 * unknown_pos      : ordered list of unknown position indices (length = unknown_count)
 * out_addrs        : pre-allocated buffer, capacity*20 bytes
 * out_indices      : pre-allocated buffer, capacity*sizeof(int64_t) bytes
 * out_count        : set to actual number of results (valid mnemonics found)
 *
 * Returns 0 on success, -1 on CUDA error.
 */
int gpu_enumerate_compute(
    int            device_id,
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
 * Upload a bloom filter to persistent GPU memory on the specified device.
 * words      : raw uint64 bitset words (BloomFilter.BitSet().Bytes())
 * word_count : number of uint64 words
 * m          : total bits (BloomFilter.Cap())
 * k          : number of hash functions (BloomFilter.K())
 * Returns 0 on success, -1 on CUDA error.
 */
int gpu_bloom_upload(int device_id, const uint64_t *words, uint64_t word_count, uint64_t m, uint32_t k);

/* Release GPU bloom filter memory for the specified device */
void gpu_bloom_free(int device_id);

/*
 * Test a single 20-byte address against the bloom filter on the specified device.
 * Returns 1 if (possibly) present, 0 if definitely absent, -1 if no filter loaded.
 */
int gpu_bloom_test_addr(int device_id, const uint8_t *addr20);

/* Release all persistent GPU enumerate buffers for the specified device (call on shutdown) */
void gpu_enumerate_cleanup(int device_id);

/* Debug: run BIP39 checksum for a single 12-word set on the specified GPU device.
 * Returns stored_cs (low 4 bits of packed bits) and sha0 (SHA256(entropy)[0]).
 * Pass condition: (sha0>>4) == stored_cs. Returns 0 on success, -1 on CUDA error. */
int gpu_bip39_debug(int device_id, const int16_t wi[12], uint8_t *stored_cs_out, uint8_t *sha0_out);

#ifdef __cplusplus
}
#endif
