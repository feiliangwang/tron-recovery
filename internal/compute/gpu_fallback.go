//go:build !cuda
// +build !cuda

package compute

import "errors"

// GPUComputer GPU计算器（非CUDA构建占位）
type GPUComputer struct{}

// NewGPUComputer 无CUDA时返回错误
func NewGPUComputer() (*GPUComputer, error) {
	return nil, errors.New("GPU not available: build with -tags cuda")
}

// Compute 未实现，返回nil
func (g *GPUComputer) Compute(mnemonics []string) [][]byte { return nil }

// Close 无操作
func (g *GPUComputer) Close() error { return nil }
