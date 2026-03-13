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
)

// GPUComputer GPU计算器（CUDA版本）
type GPUComputer struct{}

// NewGPUComputer 创建GPU计算器
func NewGPUComputer() (*GPUComputer, error) {
	n := int(C.gpu_device_count())
	if n <= 0 {
		return nil, fmt.Errorf("no CUDA-capable devices found")
	}
	return &GPUComputer{}, nil
}

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

// Close 关闭GPU计算器
func (g *GPUComputer) Close() error { return nil }
