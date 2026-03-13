package compute

import (
	"boon/internal/bip44"
	"bytes"
	"encoding/hex"
	"runtime"
	"testing"
	"time"

	"boon/internal/mnemonic"
	"boon/internal/protocol"
)

const testMnemonic = "afraid report escape reveal run sport pig blouse angry butter lock about"

func TestNewCPUComputer(t *testing.T) {
	c := NewCPUComputer()
	if c == nil {
		t.Fatal("NewCPUComputer returned nil")
	}
}

func TestCPUComputerClose(t *testing.T) {
	c := NewCPUComputer()
	if err := c.Close(); err != nil {
		t.Errorf("Close() returned error: %v", err)
	}
}

func TestCPUComputerCompute(t *testing.T) {
	c := NewCPUComputer()
	addresses := c.Compute([]string{testMnemonic})
	if len(addresses) != 1 {
		t.Fatalf("Compute returned %d addresses, want 1", len(addresses))
	}
	if len(addresses[0]) != 20 {
		t.Errorf("address length = %d, want 20", len(addresses[0]))
	}
	if hex.EncodeToString(addresses[0]) != "296e1e734897fc64d40b17dabd0ab4a812748542" {
		t.Errorf("unexpected address: %s", hex.EncodeToString(addresses[0]))
	}
	if bip44.GetTronAddress(addresses[0]) != "TDkGd84eSuvFcUXqnvNY8UDmAj1ZE6umZJ" {
		t.Errorf("unexpected TRON address: %s", bip44.GetTronAddress(addresses[0]))
	}
}

func TestCPUComputerComputeMultiple(t *testing.T) {
	c := NewCPUComputer()
	addresses := c.Compute([]string{testMnemonic, testMnemonic})
	if len(addresses) != 2 {
		t.Errorf("Compute returned %d addresses, want 2", len(addresses))
	}
	// 相同助记词应产生相同地址
	if !bytes.Equal(addresses[0], addresses[1]) {
		t.Error("same mnemonic should produce the same address")
	}
}

func TestCPUComputerDeterministic(t *testing.T) {
	c := NewCPUComputer()
	addrs1 := c.Compute([]string{testMnemonic})
	addrs2 := c.Compute([]string{testMnemonic})
	if !bytes.Equal(addrs1[0], addrs2[0]) {
		t.Error("Compute is not deterministic")
	}
}

func TestCPUComputerEmpty(t *testing.T) {
	c := NewCPUComputer()
	addresses := c.Compute([]string{})
	if len(addresses) != 0 {
		t.Errorf("Compute returned %d addresses for empty input, want 0", len(addresses))
	}
}

// ---- CompactComputer ----

func TestNewCompactComputer(t *testing.T) {
	c := NewCompactComputer(4, NewCPUComputer())
	if c == nil {
		t.Fatal("NewCompactComputer returned nil")
	}
	if c.workers != 4 {
		t.Errorf("workers = %d, want 4", c.workers)
	}
}

func TestNewCompactComputerDefaultWorkers(t *testing.T) {
	c := NewCompactComputer(0, NewCPUComputer())
	if c.workers != 4 {
		t.Errorf("workers = %d, want 4 (default)", c.workers)
	}
}

// mockEnumerator 仅对指定索引返回有效助记词
type mockEnumerator struct {
	validIdx int64
	words    []string
}

func (m *mockEnumerator) EnumerateAt(idx int64, validator *mnemonic.Validator) ([]string, bool) {
	if idx == m.validIdx {
		return m.words, true
	}
	return nil, false
}

func TestCompactComputerComputeRange(t *testing.T) {
	c := NewCompactComputer(2, NewCPUComputer())

	enum := &mockEnumerator{
		validIdx: 3,
		words:    splitMnemonic(testMnemonic),
	}
	task := &protocol.CompactTask{
		TaskID:   10,
		JobID:    1,
		StartIdx: 0,
		EndIdx:   10,
	}

	result := c.ComputeRange(enum, task, nil)

	if result.TaskID != 10 {
		t.Errorf("result.TaskID = %d, want 10", result.TaskID)
	}
	if len(result.Matches) != 1 {
		t.Fatalf("Matches count = %d, want 1", len(result.Matches))
	}
	if result.Matches[0].Index != 3 {
		t.Errorf("match index = %d, want 3", result.Matches[0].Index)
	}
	if len(result.Matches[0].Address) != 20 {
		t.Errorf("address length = %d, want 20", len(result.Matches[0].Address))
	}
}

func TestCompactComputerComputeRangeNoMatches(t *testing.T) {
	c := NewCompactComputer(2, NewCPUComputer())

	enum := &mockEnumerator{validIdx: 999, words: splitMnemonic(testMnemonic)}
	task := &protocol.CompactTask{TaskID: 5, JobID: 1, StartIdx: 0, EndIdx: 5}

	result := c.ComputeRange(enum, task, nil)
	if len(result.Matches) != 0 {
		t.Errorf("expected 0 matches, got %d", len(result.Matches))
	}
}

func TestCompactComputerBloomFilterAcceptsAll(t *testing.T) {
	c := NewCompactComputer(2, NewCPUComputer())
	enum := &mockEnumerator{validIdx: 3, words: splitMnemonic(testMnemonic)}
	task := &protocol.CompactTask{TaskID: 1, JobID: 1, StartIdx: 0, EndIdx: 10}

	result := c.ComputeRange(enum, task, func(_ []byte) bool { return true })
	if len(result.Matches) != 1 {
		t.Errorf("with accept-all bloom filter, expected 1 match, got %d", len(result.Matches))
	}
}

func TestCompactComputerBloomFilterRejectsAll(t *testing.T) {
	c := NewCompactComputer(2, NewCPUComputer())
	enum := &mockEnumerator{validIdx: 3, words: splitMnemonic(testMnemonic)}
	task := &protocol.CompactTask{TaskID: 1, JobID: 1, StartIdx: 0, EndIdx: 10}

	result := c.ComputeRange(enum, task, func(_ []byte) bool { return false })
	if len(result.Matches) != 0 {
		t.Errorf("with reject-all bloom filter, expected 0 matches, got %d", len(result.Matches))
	}
}

func TestCompactComputerDeterministic(t *testing.T) {
	c := NewCompactComputer(2, NewCPUComputer())
	enum := &mockEnumerator{validIdx: 3, words: splitMnemonic(testMnemonic)}
	task := &protocol.CompactTask{TaskID: 1, JobID: 1, StartIdx: 0, EndIdx: 10}

	r1 := c.ComputeRange(enum, task, nil)
	r2 := c.ComputeRange(enum, task, nil)

	if len(r1.Matches) != len(r2.Matches) {
		t.Error("ComputeRange is not deterministic in match count")
	}
	if len(r1.Matches) > 0 && !bytes.Equal(r1.Matches[0].Address, r2.Matches[0].Address) {
		t.Error("ComputeRange is not deterministic in address")
	}
}

func TestCompactComputerEmptyRange(t *testing.T) {
	c := NewCompactComputer(2, NewCPUComputer())
	enum := &mockEnumerator{validIdx: 0, words: splitMnemonic(testMnemonic)}
	task := &protocol.CompactTask{TaskID: 1, JobID: 1, StartIdx: 5, EndIdx: 5}

	result := c.ComputeRange(enum, task, nil)
	if len(result.Matches) != 0 {
		t.Errorf("empty range should produce 0 matches, got %d", len(result.Matches))
	}
}

func splitMnemonic(m string) []string {
	words := make([]string, 0, 12)
	word := ""
	for _, ch := range m + " " {
		if ch == ' ' {
			if word != "" {
				words = append(words, word)
				word = ""
			}
		} else {
			word += string(ch)
		}
	}
	return words
}

// alwaysValidEnumerator 任何索引都返回同一有效助记词（用于纯加密吞吐量基准）
type alwaysValidEnumerator struct {
	words []string
}

func (e *alwaysValidEnumerator) EnumerateAt(_ int64, _ *mnemonic.Validator) ([]string, bool) {
	return e.words, true
}

// BenchmarkComputeAddress 单次地址计算（BIP39种子 + BIP44派生 + Keccak256）
func BenchmarkComputeAddress(b *testing.B) {
	c := NewCPUComputer()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c.computeOne(testMnemonic)
	}
}

// BenchmarkCompactComputerComputeRange 多线程批量计算吞吐量
func BenchmarkCompactComputerComputeRange(b *testing.B) {
	workers := runtime.NumCPU()
	c := NewCompactComputer(workers, NewCPUComputer())
	enum := &alwaysValidEnumerator{words: splitMnemonic(testMnemonic)}
	const N = int64(4000) // 4000/500 = 8个子批次 → 充分利用8核并行

	b.ReportAllocs()
	b.ResetTimer()
	start := time.Now()
	total := int64(0)
	for i := 0; i < b.N; i++ {
		task := &protocol.CompactTask{TaskID: 1, JobID: 1, StartIdx: 0, EndIdx: N}
		c.ComputeRange(enum, task, nil)
		total += N
	}
	elapsed := time.Since(start)
	b.ReportMetric(float64(total)/elapsed.Seconds(), "indices/s")
}
