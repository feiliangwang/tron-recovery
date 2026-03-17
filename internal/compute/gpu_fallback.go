//go:build !cuda
// +build !cuda

package compute

import "errors"

// GPUComputer GPU计算器（非CUDA构建占位）
type GPUComputer struct{}

// GPUDeviceCount 非CUDA构建始终返回0
func GPUDeviceCount() int { return 0 }

// NewGPUComputer 无CUDA时返回错误
func NewGPUComputer(deviceID int) (*GPUComputer, error) {
	return nil, errors.New("GPU not available: build with -tags cuda")
}

// NewGPUComputerAll 无CUDA时返回错误
func NewGPUComputerAll() ([]*GPUComputer, error) {
	return nil, errors.New("GPU not available: build with -tags cuda")
}

// DeviceID 非CUDA构建始终返回0
func (g *GPUComputer) DeviceID() int { return 0 }

// Compute 未实现，返回nil
func (g *GPUComputer) Compute(mnemonics []string) [][]byte { return nil }

// Close 无操作
func (g *GPUComputer) Close() error { return nil }
