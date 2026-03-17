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
	deviceID   int
	bloomOnGPU bool
	flatBuf    []byte
	offsetBuf  []C.int
	lengthBuf  []C.int
	unkBuf     []C.int8_t
}

func supportedGPUDeviceCount() int {
	n := int(C.gpu_device_count())
	maxSlots := int(C.gpu_max_devices())
	if n < maxSlots {
		return n
	}
	return maxSlots
}

// GPUDeviceCount 返回可用 CUDA 设备数量
func GPUDeviceCount() int {
	return supportedGPUDeviceCount()
}

// NewGPUComputer 创建指定设备的GPU计算器
func NewGPUComputer(deviceID int) (*GPUComputer, error) {
	n := supportedGPUDeviceCount()
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
	n := supportedGPUDeviceCount()
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

func (g *GPUComputer) flattenMnemonics(mnemonics []string) (flat []byte, offsets []C.int, lengths []C.int) {
	count := len(mnemonics)
	totalBytes := 0
	for _, m := range mnemonics {
		totalBytes += len(m)
	}
	if cap(g.flatBuf) < totalBytes {
		g.flatBuf = make([]byte, totalBytes)
	} else {
		g.flatBuf = g.flatBuf[:totalBytes]
	}
	flat = g.flatBuf

	if cap(g.offsetBuf) < count {
		g.offsetBuf = make([]C.int, count)
	} else {
		g.offsetBuf = g.offsetBuf[:count]
	}
	offsets = g.offsetBuf

	if cap(g.lengthBuf) < count {
		g.lengthBuf = make([]C.int, count)
	} else {
		g.lengthBuf = g.lengthBuf[:count]
	}
	lengths = g.lengthBuf

	pos := 0
	for i, m := range mnemonics {
		offsets[i] = C.int(pos)
		lengths[i] = C.int(len(m))
		copy(flat[pos:], m)
		pos += len(m)
	}
	if len(flat) == 0 {
		flat = []byte{0}
	}
	return flat, offsets, lengths
}

// Compute 计算助记词对应的TRON地址（GPU版本）
func (g *GPUComputer) Compute(mnemonics []string) [][]byte {
	if len(mnemonics) == 0 {
		return nil
	}
	count := len(mnemonics)
	flat, offsets, lengths := g.flattenMnemonics(mnemonics)

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

	return addressViews(addrBuf, count)
}

// ComputePBKDF2Seeds computes PBKDF2-HMAC-SHA512(2048) seeds for the given
// mnemonics on GPU and returns them as a contiguous count*64-byte buffer.
func (g *GPUComputer) ComputePBKDF2Seeds(mnemonics []string) []byte {
	if len(mnemonics) == 0 {
		return nil
	}
	count := len(mnemonics)
	flat, offsets, lengths := g.flattenMnemonics(mnemonics)
	seedBuf := make([]byte, count*64)

	ret := C.gpu_compute_pbkdf2_seeds(
		C.int(g.deviceID),
		(*C.uint8_t)(unsafe.Pointer(&flat[0])),
		(*C.int)(unsafe.Pointer(&offsets[0])),
		(*C.int)(unsafe.Pointer(&lengths[0])),
		C.int(count),
		(*C.uint8_t)(unsafe.Pointer(&seedBuf[0])),
	)
	if int(ret) < 0 {
		return nil
	}
	return seedBuf
}

// BenchmarkPBKDF2Kernel measures only the GPU PBKDF2 kernel time on a
// pre-uploaded mnemonic batch. The timed region excludes Go-side flattening and
// host/device copies after the initial upload.
func (g *GPUComputer) BenchmarkPBKDF2Kernel(mnemonics []string, rounds int) (kernelMs float64, sample uint64, ok bool) {
	if len(mnemonics) == 0 || rounds <= 0 {
		return 0, 0, false
	}
	count := len(mnemonics)
	flat, offsets, lengths := g.flattenMnemonics(mnemonics)

	var ms C.float
	var sampleC C.uint64_t
	ret := C.gpu_benchmark_pbkdf2_kernel(
		C.int(g.deviceID),
		(*C.uint8_t)(unsafe.Pointer(&flat[0])),
		(*C.int)(unsafe.Pointer(&offsets[0])),
		(*C.int)(unsafe.Pointer(&lengths[0])),
		C.int(count),
		C.int(rounds),
		&ms,
		&sampleC,
	)
	if int(ret) < 0 {
		return 0, 0, false
	}
	return float64(ms), uint64(sampleC), true
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

	var knownC [12]C.int16_t
	for i, v := range knownWordIndices {
		knownC[i] = C.int16_t(v)
	}
	if cap(g.unkBuf) < len(unknownPositions) {
		g.unkBuf = make([]C.int8_t, len(unknownPositions))
	} else {
		g.unkBuf = g.unkBuf[:len(unknownPositions)]
	}
	unkC := g.unkBuf
	for i, v := range unknownPositions {
		unkC[i] = C.int8_t(v)
	}
	var unkPtr *C.int8_t
	if len(unkC) > 0 {
		unkPtr = (*C.int8_t)(unsafe.Pointer(&unkC[0]))
	}

	outAddrs := make([]byte, capacity*20)
	outIdxs := make([]int64, capacity)
	var outCount C.int

	ret := C.gpu_enumerate_compute(
		C.int(g.deviceID),
		C.int64_t(startIdx),
		C.int64_t(endIdx),
		(*C.int16_t)(unsafe.Pointer(&knownC[0])),
		unkPtr,
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
	return outIdxs[:cnt:cnt], addressViews(outAddrs, cnt), nil
}

// BloomFilterOnGPU reports whether this GPU computer currently has a bloom
// filter uploaded and can therefore filter enumerate results on-device.
func (g *GPUComputer) BloomFilterOnGPU() bool {
	return g.bloomOnGPU
}

func addressViews(buf []byte, count int) [][]byte {
	if count <= 0 {
		return nil
	}
	result := make([][]byte, count)
	for i := 0; i < count; i++ {
		start := i * 20
		end := start + 20
		result[i] = buf[start:end:end]
	}
	return result
}

// UploadBloomFilter uploads a bloom filter to persistent GPU memory on this device.
// Once uploaded, gpu_enumerate_compute will filter addresses on the GPU,
// returning only bloom-matching results and eliminating GPU→CPU address transfer.
func (g *GPUComputer) UploadBloomFilter(f *bloom.Filter) error {
	words, m, k := f.RawBits()
	if len(words) == 0 {
		return g.ClearBloomFilter()
	}
	ret := C.gpu_bloom_upload(
		C.int(g.deviceID),
		(*C.uint64_t)(unsafe.Pointer(&words[0])),
		C.uint64_t(len(words)),
		C.uint64_t(m),
		C.uint32_t(k),
	)
	if int(ret) < 0 {
		g.bloomOnGPU = false
		return fmt.Errorf("gpu_bloom_upload failed (device %d)", g.deviceID)
	}
	g.bloomOnGPU = true
	return nil
}

// ClearBloomFilter releases the currently uploaded bloom filter on this device.
func (g *GPUComputer) ClearBloomFilter() error {
	C.gpu_bloom_free(C.int(g.deviceID))
	g.bloomOnGPU = false
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
	g.bloomOnGPU = false
	return nil
}
