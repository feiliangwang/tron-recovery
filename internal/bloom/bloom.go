package bloom

import (
	"encoding/gob"
	"os"

	"github.com/bits-and-blooms/bloom/v3"
)

// Filter Bloom过滤器包装
type Filter struct {
	filter *bloom.BloomFilter
}

// NewFilter 创建Bloom过滤器
func NewFilter(expectedItems uint, falsePositiveRate float64) *Filter {
	return &Filter{
		filter: bloom.NewWithEstimates(expectedItems, falsePositiveRate),
	}
}

// LoadFromFile 从文件加载Bloom过滤器（gob格式）
func LoadFromFile(filename string) (*Filter, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	bf := bloom.New(0, 0) // 创建空的过滤器
	decoder := gob.NewDecoder(file)
	if err := decoder.Decode(bf); err != nil {
		return nil, err
	}

	return &Filter{filter: bf}, nil
}

// SaveToFile 保存Bloom过滤器到文件（gob格式）
func (f *Filter) SaveToFile(filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)
	return encoder.Encode(f.filter)
}

// Add 添加数据到过滤器
func (f *Filter) Add(data []byte) {
	f.filter.Add(data)
}

// Contains 检查数据是否可能在过滤器中
func (f *Filter) Contains(data []byte) bool {
	return f.filter.Test(data)
}

// TestAndAdd 测试并添加
func (f *Filter) TestAndAdd(data []byte) bool {
	return f.filter.TestAndAdd(data)
}

// Clear 清空过滤器
func (f *Filter) Clear() {
	f.filter.ClearAll()
}
