//go:build cuda
// +build cuda

package compute

/*
#cgo LDFLAGS: -L${SRCDIR} -lgpu_cuda -L/usr/local/cuda/lib64 -lcudart
#include "gpu_bridge.h"
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"unsafe"

	"boon/internal/bloom"
)

// GPUComputer GPU计算器（CUDA版本）
type GPUComputer struct {
	deviceID int
}

// GPUDeviceCount 返回可用 CUDA 设备数量
func GPUDeviceCount() int {
	return int(C.gpu_device_count())
}

// NewGPUComputer 创建指定设备的GPU计算器
func NewGPUComputer(deviceID int) (*GPUComputer, error) {
	n := int(C.gpu_device_count())
	if n <= 0 {
		return nil, fmt.Errorf("no CUDA-capable devices found")
	}
	if deviceID < 0 || deviceID >= n {
		return nil, fmt.Errorf("device %d not available (found %d device(s))", deviceID, n)
	}
	return &GPUComputer{deviceID: deviceID}, nil
}

// NewGPUComputerAll 创建所有可用 GPU 设备的计算器列表
func NewGPUComputerAll() ([]*GPUComputer, error) {
	n := int(C.gpu_device_count())
	if n <= 0 {
		return nil, fmt.Errorf("no CUDA-capable devices found")
	}
	result := make([]*GPUComputer, n)
	for i := range result {
		result[i] = &GPUComputer{deviceID: i}
	}
	return result, nil
}

// DeviceID 返回该计算器绑定的 GPU 设备 ID
func (g *GPUComputer) DeviceID() int { return g.deviceID }

// Compute 计算助记词对应的TRON地址（GPU版本）
func (g *GPUComputer) Compute(mnemonics []string) [][]byte {
	if len(mnemonics) == 0 {
		return nil
	}
	count := len(mnemonics)

	// Flatten mnemonics into a single byte buffer
	var flat []byte
	offsets := make([]C.int, count)
	lengths := make([]C.int, count)
	for i, m := range mnemonics {
		offsets[i] = C.int(len(flat))
		lengths[i] = C.int(len(m))
		flat = append(flat, []byte(m)...)
	}
	if len(flat) == 0 {
		flat = []byte{0}
	}

	addrBuf := make([]byte, count*20)

	ret := C.gpu_compute_addresses(
		C.int(g.deviceID),
		(*C.uint8_t)(unsafe.Pointer(&flat[0])),
		(*C.int)(unsafe.Pointer(&offsets[0])),
		(*C.int)(unsafe.Pointer(&lengths[0])),
		C.int(count),
		(*C.uint8_t)(unsafe.Pointer(&addrBuf[0])),
	)

	if int(ret) < 0 {
		// Fall back to CPU on GPU error
		cpu := NewCPUComputer()
		return cpu.Compute(mnemonics)
	}

	result := make([][]byte, count)
	for i := range result {
		addr := make([]byte, 20)
		copy(addr, addrBuf[i*20:(i+1)*20])
		result[i] = addr
	}
	return result
}

// EnumerateCompute performs BIP39 enumeration and TRON address derivation entirely on the GPU.
// knownWordIndices: 12 entries where -1 means unknown position (word index 0-2047 otherwise).
// unknownPositions: position indices in enumeration order (same order as index decomposition).
// Returns matched (indices, addresses) pairs that passed BIP39 checksum validation.
func (g *GPUComputer) EnumerateCompute(
	startIdx, endIdx int64,
	knownWordIndices []int16,
	unknownPositions []int8,
) (indices []int64, addresses [][]byte, err error) {
	total := endIdx - startIdx
	if total <= 0 {
		return nil, nil, nil
	}

	// Capacity: pass rate ~6.2% for 4 unknowns; allocate 15% for safety margin
	capacity := int(total/7) + 2048
	if capacity < 2048 {
		capacity = 2048
	}

	knownC := make([]C.int16_t, 12)
	for i, v := range knownWordIndices {
		knownC[i] = C.int16_t(v)
	}
	unkC := make([]C.int8_t, len(unknownPositions))
	for i, v := range unknownPositions {
		unkC[i] = C.int8_t(v)
	}

	outAddrs := make([]byte, capacity*20)
	outIdxs := make([]int64, capacity)
	var outCount C.int

	ret := C.gpu_enumerate_compute(
		C.int(g.deviceID),
		C.int64_t(startIdx),
		C.int64_t(endIdx),
		(*C.int16_t)(unsafe.Pointer(&knownC[0])),
		(*C.int8_t)(unsafe.Pointer(&unkC[0])),
		C.int8_t(len(unknownPositions)),
		(*C.uint8_t)(unsafe.Pointer(&outAddrs[0])),
		(*C.int64_t)(unsafe.Pointer(&outIdxs[0])),
		C.int(capacity),
		&outCount,
	)
	if int(ret) < 0 {
		return nil, nil, fmt.Errorf("gpu_enumerate_compute failed (device %d)", g.deviceID)
	}

	cnt := int(outCount)
	if cnt > capacity {
		cnt = capacity
	}
	addresses = make([][]byte, cnt)
	for i := range addresses {
		addr := make([]byte, 20)
		copy(addr, outAddrs[i*20:(i+1)*20])
		addresses[i] = addr
	}
	return outIdxs[:cnt], addresses, nil
}

// UploadBloomFilter uploads a bloom filter to persistent GPU memory on this device.
// Once uploaded, gpu_enumerate_compute will filter addresses on the GPU,
// returning only bloom-matching results and eliminating GPU→CPU address transfer.
func (g *GPUComputer) UploadBloomFilter(f *bloom.Filter) error {
	words, m, k := f.RawBits()
	if len(words) == 0 {
		return nil
	}
	ret := C.gpu_bloom_upload(
		C.int(g.deviceID),
		(*C.uint64_t)(unsafe.Pointer(&words[0])),
		C.uint64_t(len(words)),
		C.uint64_t(m),
		C.uint32_t(k),
	)
	if int(ret) < 0 {
		return fmt.Errorf("gpu_bloom_upload failed (device %d)", g.deviceID)
	}
	return nil
}

// BloomTestAddr tests a 20-byte address against the currently-uploaded GPU bloom filter.
// Returns true if present, false if absent. Panics if no filter is loaded.
func (g *GPUComputer) BloomTestAddr(addr []byte) bool {
	if len(addr) != 20 {
		panic("BloomTestAddr: address must be 20 bytes")
	}
	ret := C.gpu_bloom_test_addr(C.int(g.deviceID), (*C.uint8_t)(unsafe.Pointer(&addr[0])))
	return int(ret) == 1
}

// BIP39Debug runs the BIP39 checksum validation for a single 12-word set on GPU.
// Returns (stored_cs, sha0, err). Pass condition: sha0>>4 == stored_cs.
func (g *GPUComputer) BIP39Debug(wordIndices [12]int16) (storedCS, sha0 byte, err error) {
	wiC := make([]C.int16_t, 12)
	for i, v := range wordIndices {
		wiC[i] = C.int16_t(v)
	}
	var cStored, cSha C.uint8_t
	ret := C.gpu_bip39_debug(
		C.int(g.deviceID),
		(*C.int16_t)(unsafe.Pointer(&wiC[0])),
		&cStored, &cSha,
	)
	if int(ret) < 0 {
		return 0, 0, fmt.Errorf("gpu_bip39_debug failed (device %d)", g.deviceID)
	}
	return byte(cStored), byte(cSha), nil
}

// Close 关闭GPU计算器，释放持久化GPU内存
func (g *GPUComputer) Close() error {
	C.gpu_enumerate_cleanup(C.int(g.deviceID))
	return nil
}
