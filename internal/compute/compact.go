package compute

import (
	"strings"
	"sync"

	"boon/internal/mnemonic"
	"boon/internal/protocol"
)

// Enumerator 枚举器接口
type Enumerator interface {
	EnumerateAt(idx int64, validator *mnemonic.Validator) ([]string, bool)
}

// CompactComputer 紧凑计算器
type CompactComputer struct {
	workers  int
	computer SeedComputer
}

// NewCompactComputer 创建紧凑计算器
func NewCompactComputer(workers int, computer SeedComputer) *CompactComputer {
	if workers <= 0 {
		workers = 4
	}
	return &CompactComputer{workers: workers, computer: computer}
}

// ComputeRange 计算位置范围内的匹配
func (c *CompactComputer) ComputeRange(
	enum Enumerator,
	task *protocol.CompactTask,
	bloomFilter func([]byte) bool,
) *protocol.CompactResult {
	result := &protocol.CompactResult{
		TaskID:  task.TaskID,
		Matches: make([]protocol.MatchData, 0),
	}

	var mu sync.Mutex
	var wg sync.WaitGroup
	sem := make(chan struct{}, c.workers)

	// 按批次并行处理
	batchSize := int64(500)
	for start := task.StartIdx; start < task.EndIdx; start += batchSize {
		end := start + batchSize
		if end > task.EndIdx {
			end = task.EndIdx
		}

		wg.Add(1)
		go func(s, e int64) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			matches := c.processBatch(enum, s, e, bloomFilter)
			if len(matches) > 0 {
				mu.Lock()
				result.Matches = append(result.Matches, matches...)
				mu.Unlock()
			}
		}(start, end)
	}

	wg.Wait()
	return result
}

// processBatch 处理一个批次
func (c *CompactComputer) processBatch(
	enum Enumerator,
	start, end int64,
	bloomFilter func([]byte) bool,
) []protocol.MatchData {
	matches := make([]protocol.MatchData, 0)
	validator := mnemonic.NewValidator()
	idxMap := make(map[string]int64)
	mnemonics := make([]string, 0)
	for idx := start; idx < end; idx++ {
		// 枚举位置
		words, valid := enum.EnumerateAt(idx, validator)
		if !valid {
			continue
		}
		join := strings.Join(words, " ")
		idxMap[join] = idx
		mnemonics = append(mnemonics, join)
	}
	// 计算地址
	for idx, address := range c.computer.Compute(mnemonics) {
		// Bloom过滤（如果有）
		if bloomFilter != nil && !bloomFilter(address) {
			continue
		}
		// 匹配
		matches = append(matches, protocol.MatchData{
			Index:   idxMap[mnemonics[idx]],
			Address: address,
		})
	}

	return matches
}
