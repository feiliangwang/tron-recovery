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

// IndexedEnumerator 可将模板以词索引形式提供给 GPU 侧枚举（可选接口）
type IndexedEnumerator interface {
	TemplateIndices() (knownWordIndices []int16, unknownPositions []int8)
}

// rangeEnumerator 支持 GPU 原生枚举+计算（可选接口，由 GPUComputer 实现）
type rangeEnumerator interface {
	EnumerateCompute(startIdx, endIdx int64, knownWordIndices []int16, unknownPositions []int8) ([]int64, [][]byte, error)
}

// streamedRangeEnumerator is optionally implemented by GPUComputer to enable
// double-buffered CUDA stream pipelining: filter[N+1] overlaps derive[N].
type streamedRangeEnumerator interface {
	EnumerateComputeLaunch(streamIdx int, startIdx, endIdx int64,
		knownWordIndices []int16, unknownPositions []int8, capacity int,
	) (prevIndices []int64, prevAddresses [][]byte, err error)
	EnumerateComputeFlush(streamIdx int, capacity int) (indices []int64, addresses [][]byte, err error)
}

type gpuBloomEnumerator interface {
	BloomFilterOnGPU() bool
}

// CompactComputer 紧凑计算器
type CompactComputer struct {
	workers     int
	enumWorkers int // 枚举并发数（>1 时启用流水线模式）
	batchSize   int64
	computer    SeedComputer
}

// NewCompactComputer 创建紧凑计算器
func NewCompactComputer(workers int, computer SeedComputer) *CompactComputer {
	if workers <= 0 {
		workers = 4
	}
	return &CompactComputer{workers: workers, batchSize: 500, computer: computer}
}

// SetBatchSize 设置每次 Compute 调用的批次大小
func (c *CompactComputer) SetBatchSize(n int64) {
	if n > 0 {
		c.batchSize = n
	}
}

// SetEnumWorkers 设置枚举并发数（>1 时启用流水线：多CPU枚举→单GPU计算）
func (c *CompactComputer) SetEnumWorkers(n int) {
	if n > 0 {
		c.enumWorkers = n
	}
}

// GetSeedComputer 返回底层 SeedComputer（供外层检查是否支持 GPU bloom 上传等接口）
func (c *CompactComputer) GetSeedComputer() SeedComputer {
	return c.computer
}

// enumBatch 枚举结果批次
type enumBatch struct {
	mnemonics []string
	idxMap    map[string]int64
}

// ComputeRange 计算位置范围内的匹配
func (c *CompactComputer) ComputeRange(
	enum Enumerator,
	task *protocol.CompactTask,
	bloomFilter func([]byte) bool,
) *protocol.CompactResult {
	// GPU原生路径优先：两阶段kernel（bip39_filter_kernel + tron_derive_kernel）
	// 消除了SIMT warp分歧，100%线程利用率，CPU→GPU传输仅56字节/任务
	if re, ok := c.computer.(rangeEnumerator); ok {
		if ie, ok := enum.(IndexedEnumerator); ok {
			return c.computeRangeGPUNative(re, enum, ie, task, bloomFilter)
		}
	}
	// 流水线模式：多CPU并行枚举 → GPU批量计算（无GPU原生支持时的后备）
	if c.enumWorkers > 1 {
		return c.computeRangePipelined(enum, task, bloomFilter)
	}
	return c.computeRangeParallel(enum, task, bloomFilter)
}

// computeRangeGPUNative GPU 原生枚举路径：索引范围直接送入 GPU，完全跳过 CPU 枚举
func (c *CompactComputer) computeRangeGPUNative(
	re rangeEnumerator,
	enum Enumerator,
	ie IndexedEnumerator,
	task *protocol.CompactTask,
	bloomFilter func([]byte) bool,
) *protocol.CompactResult {
	result := &protocol.CompactResult{
		TaskID:  task.TaskID,
		Matches: make([]protocol.MatchData, 0),
	}

	gpuBloomActive := false
	if bloomFilter != nil {
		if gbe, ok := c.computer.(gpuBloomEnumerator); ok && gbe.BloomFilterOnGPU() {
			gpuBloomActive = true
		}
	}

	known, unkPos := ie.TemplateIndices()
	batchSize := c.batchSize
	capacity := int(batchSize/7) + 2048

	processMatch := func(idxs []int64, addrs [][]byte) {
		for i, addr := range addrs {
			if !gpuBloomActive && bloomFilter != nil && !bloomFilter(addr) {
				continue
			}
			result.Matches = append(result.Matches, protocol.MatchData{
				Index:   idxs[i],
				Address: addr,
			})
		}
	}

	// Streaming path: overlap tron_derive[N] with bip39_filter[N+1] via CUDA streams
	if sre, ok := c.computer.(streamedRangeEnumerator); ok {
		streamIdx := 0
		streamOK := true

		for start := task.StartIdx; start < task.EndIdx && streamOK; start += batchSize {
			end := start + batchSize
			if end > task.EndIdx {
				end = task.EndIdx
			}

			prevIdxs, prevAddrs, err := sre.EnumerateComputeLaunch(streamIdx, start, end, known, unkPos, capacity)
			if err != nil {
				// Flush any pending GPU work then fall back to blocking path
				sre.EnumerateComputeFlush(streamIdx^1, capacity) //nolint:errcheck
				streamOK = false
				// Fall back to blocking EnumerateCompute for this and remaining batches
				for s := start; s < task.EndIdx; s += batchSize {
					e := s + batchSize
					if e > task.EndIdx {
						e = task.EndIdx
					}
					idxs, addrs, gerr := re.EnumerateCompute(s, e, known, unkPos)
					if gerr != nil {
						eb := c.enumerateBatch(enum, s, e)
						cpuAddrs := c.computer.Compute(eb.mnemonics)
						for i, addr := range cpuAddrs {
							if bloomFilter != nil && !bloomFilter(addr) {
								continue
							}
							result.Matches = append(result.Matches, protocol.MatchData{
								Index: eb.idxMap[eb.mnemonics[i]], Address: addr,
							})
						}
						continue
					}
					processMatch(idxs, addrs)
				}
				return result
			}
			processMatch(prevIdxs, prevAddrs)
			streamIdx ^= 1
		}

		if streamOK {
			// Flush last pending batch (stream used in the last launch = streamIdx^1)
			lastIdxs, lastAddrs, _ := sre.EnumerateComputeFlush(streamIdx^1, capacity)
			processMatch(lastIdxs, lastAddrs)
		}
		return result
	}

	// Non-streaming (blocking) fallback
	for start := task.StartIdx; start < task.EndIdx; start += batchSize {
		end := start + batchSize
		if end > task.EndIdx {
			end = task.EndIdx
		}

		idxs, addrs, err := re.EnumerateCompute(start, end, known, unkPos)
		if err != nil {
			// GPU 失败时回退到 CPU 流水线处理本批
			eb := c.enumerateBatch(enum, start, end)
			cpuAddrs := c.computer.Compute(eb.mnemonics)
			for i, addr := range cpuAddrs {
				if bloomFilter != nil && !bloomFilter(addr) {
					continue
				}
				result.Matches = append(result.Matches, protocol.MatchData{
					Index:   eb.idxMap[eb.mnemonics[i]],
					Address: addr,
				})
			}
			continue
		}

		processMatch(idxs, addrs)
	}

	return result
}

// computeRangePipelined 流水线模式：多CPU并行枚举 → 单线程计算（适合GPU）
// 枚举和计算重叠执行，消除CPU等GPU / GPU等CPU的空闲时间
func (c *CompactComputer) computeRangePipelined(
	enum Enumerator,
	task *protocol.CompactTask,
	bloomFilter func([]byte) bool,
) *protocol.CompactResult {
	result := &protocol.CompactResult{
		TaskID:  task.TaskID,
		Matches: make([]protocol.MatchData, 0),
	}

	batchSize := c.batchSize
	// channel buffer = enumWorkers*2，让枚举提前预备，避免 GPU 空等
	batchCh := make(chan *enumBatch, c.enumWorkers*2)

	// 枚举生产者：enumWorkers 个并行 goroutine
	go func() {
		var wg sync.WaitGroup
		sem := make(chan struct{}, c.enumWorkers)
		for start := task.StartIdx; start < task.EndIdx; start += batchSize {
			end := start + batchSize
			if end > task.EndIdx {
				end = task.EndIdx
			}
			wg.Add(1)
			sem <- struct{}{}
			go func(s, e int64) {
				defer wg.Done()
				defer func() { <-sem }()
				batchCh <- c.enumerateBatch(enum, s, e)
			}(start, end)
		}
		wg.Wait()
		close(batchCh)
	}()

	// 计算消费者：单线程顺序消费（GPU 不支持并发调用）
	for eb := range batchCh {
		if len(eb.mnemonics) == 0 {
			continue
		}
		addrs := c.computer.Compute(eb.mnemonics)
		for i, addr := range addrs {
			if bloomFilter != nil && !bloomFilter(addr) {
				continue
			}
			result.Matches = append(result.Matches, protocol.MatchData{
				Index:   eb.idxMap[eb.mnemonics[i]],
				Address: addr,
			})
		}
	}

	return result
}

// computeRangeParallel 原始并行模式：每个 goroutine 独立枚举+计算（适合多核CPU）
func (c *CompactComputer) computeRangeParallel(
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

	batchSize := c.batchSize
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

			eb := c.enumerateBatch(enum, s, e)
			addrs := c.computer.Compute(eb.mnemonics)
			matches := make([]protocol.MatchData, 0)
			for i, addr := range addrs {
				if bloomFilter != nil && !bloomFilter(addr) {
					continue
				}
				matches = append(matches, protocol.MatchData{
					Index:   eb.idxMap[eb.mnemonics[i]],
					Address: addr,
				})
			}
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

// enumerateBatch 枚举一个批次，返回有效助记词和索引映射
func (c *CompactComputer) enumerateBatch(enum Enumerator, start, end int64) *enumBatch {
	validator := mnemonic.NewValidator()
	idxMap := make(map[string]int64, int(end-start)/16)
	mnemonics := make([]string, 0, int(end-start)/16)
	for idx := start; idx < end; idx++ {
		words, valid := enum.EnumerateAt(idx, validator)
		if !valid {
			continue
		}
		join := strings.Join(words, " ")
		idxMap[join] = idx
		mnemonics = append(mnemonics, join)
	}
	return &enumBatch{mnemonics: mnemonics, idxMap: idxMap}
}
