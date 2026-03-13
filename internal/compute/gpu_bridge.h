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

#ifdef __cplusplus
}
#endif
