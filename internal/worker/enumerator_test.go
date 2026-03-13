package worker

import (
	"runtime"
	"testing"
	"time"

	"boon/internal/compute"
	"boon/internal/mnemonic"
	"boon/internal/protocol"
)

// "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about"
// is a known-valid BIP39 mnemonic. "about" is WordList[3].
func makeAbandonTemplate() *TaskTemplate {
	return &TaskTemplate{
		JobID: 1,
		Words: []string{
			"abandon", "abandon", "abandon", "abandon", "abandon", "abandon",
			"abandon", "abandon", "abandon", "abandon", "abandon", "",
		},
		UnknownPos: []int{11},
	}
}

func TestNewLocalEnumerator(t *testing.T) {
	tmpl := makeAbandonTemplate()
	e := NewLocalEnumerator(tmpl)
	if e == nil {
		t.Fatal("NewLocalEnumerator returned nil")
	}
}

func TestNewLocalEnumeratorInfersUnknownPos(t *testing.T) {
	// Create template without explicit UnknownPos (empty slice)
	tmpl := &TaskTemplate{
		JobID: 1,
		Words: []string{
			"", "abandon", "abandon", "abandon", "abandon", "abandon",
			"abandon", "abandon", "abandon", "abandon", "abandon", "abandon",
		},
		UnknownPos: []int{},
	}
	e := NewLocalEnumerator(tmpl)
	if len(e.template.UnknownPos) != 1 || e.template.UnknownPos[0] != 0 {
		t.Errorf("InferUnknownPos: got %v, want [0]", e.template.UnknownPos)
	}
}

func TestGetWordCount(t *testing.T) {
	e := NewLocalEnumerator(makeAbandonTemplate())
	if e.GetWordCount() != 2048 {
		t.Errorf("GetWordCount() = %d, want 2048", e.GetWordCount())
	}
}

func TestEnumerateAtKnownValid(t *testing.T) {
	e := NewLocalEnumerator(makeAbandonTemplate())
	v := mnemonic.NewValidator()

	// "about" is mnemonic.WordList[3], so index 3 should produce the known-valid mnemonic
	words, ok := e.EnumerateAt(3, v)
	if !ok {
		t.Fatal("EnumerateAt(3) should return valid for the known-valid mnemonic")
	}
	if len(words) != 12 {
		t.Fatalf("EnumerateAt returned %d words, want 12", len(words))
	}
	if words[11] != "about" {
		t.Errorf("words[11] = %q, want %q", words[11], "about")
	}
	// First 11 words should all be "abandon"
	for i := 0; i < 11; i++ {
		if words[i] != "abandon" {
			t.Errorf("words[%d] = %q, want %q", i, words[i], "abandon")
		}
	}
}

func TestEnumerateAtInvalidChecksum(t *testing.T) {
	e := NewLocalEnumerator(makeAbandonTemplate())
	v := mnemonic.NewValidator()

	// Index 0 → "abandon" at position 11 → all 12 "abandon" words → invalid checksum
	_, ok := e.EnumerateAt(0, v)
	if ok {
		t.Error("EnumerateAt(0) should return false (invalid BIP39 checksum)")
	}
}

func TestEnumerateAtDifferentIndicesProduceDifferentWords(t *testing.T) {
	e := NewLocalEnumerator(makeAbandonTemplate())
	v := mnemonic.NewValidator()

	// Find two valid indices and ensure they differ
	var valid []int64
	for idx := int64(0); idx < 2048 && len(valid) < 2; idx++ {
		if _, ok := e.EnumerateAt(idx, v); ok {
			valid = append(valid, idx)
		}
	}
	if len(valid) < 2 {
		t.Skip("fewer than 2 valid indices found; skipping comparison test")
	}

	w1, _ := e.EnumerateAt(valid[0], v)
	w2, _ := e.EnumerateAt(valid[1], v)
	if w1[11] == w2[11] {
		t.Error("different indices should produce different words")
	}
}

func TestEnumerateAtMultipleUnknown(t *testing.T) {
	// Two unknown positions
	tmpl := &TaskTemplate{
		JobID: 2,
		Words: []string{
			"", "abandon", "abandon", "abandon", "abandon", "abandon",
			"abandon", "abandon", "abandon", "abandon", "abandon", "",
		},
		UnknownPos: []int{0, 11},
	}
	e := NewLocalEnumerator(tmpl)
	v := mnemonic.NewValidator()

	// Index 0 → both positions get WordList[0]="abandon" → invalid checksum
	// Scan a small range to find a valid one
	found := false
	for idx := int64(0); idx < 1000; idx++ {
		if words, ok := e.EnumerateAt(idx, v); ok {
			if len(words) != 12 {
				t.Errorf("words length = %d, want 12", len(words))
			}
			found = true
			break
		}
	}
	if !found {
		t.Log("no valid mnemonic found in first 1000 indices for 2-unknown template (may be expected)")
	}
}

func TestGetMnemonic(t *testing.T) {
	e := NewLocalEnumerator(makeAbandonTemplate())
	words := []string{"word1", "word2", "word3"}
	got := e.GetMnemonic(words)
	want := "word1 word2 word3"
	if got != want {
		t.Errorf("GetMnemonic() = %q, want %q", got, want)
	}
}

func TestGetMnemonicEmpty(t *testing.T) {
	e := NewLocalEnumerator(makeAbandonTemplate())
	got := e.GetMnemonic([]string{})
	if got != "" {
		t.Errorf("GetMnemonic([]) = %q, want %q", got, "")
	}
}

// BenchmarkEnumerateAt 单个索引处理速度（含BIP39校验和验证）
func BenchmarkEnumerateAt(b *testing.B) {
	e := NewLocalEnumerator(makeAbandonTemplate())
	v := mnemonic.NewValidator()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		e.EnumerateAt(int64(i%2048), v)
	}
}

// BenchmarkComputeRangeFull 完整流水线吞吐量（BIP39校验和 + PBKDF2 + BIP44 + Keccak256）
func BenchmarkComputeRangeFull(b *testing.B) {
	workers := runtime.NumCPU()
	c := compute.NewCompactComputer(workers, compute.NewCPUComputer())
	enum := NewLocalEnumerator(makeAbandonTemplate())
	const N = int64(2048) // 1个未知词 = 2048种组合

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

func TestEnumerateAtIndexMapping(t *testing.T) {
	// Template with 1 unknown at position 0
	tmpl := &TaskTemplate{
		JobID: 1,
		Words: []string{
			"", "abandon", "abandon", "abandon", "abandon", "abandon",
			"abandon", "abandon", "abandon", "abandon", "abandon", "about",
		},
		UnknownPos: []int{0},
	}
	e := NewLocalEnumerator(tmpl)
	v := mnemonic.NewValidator()

	// WordList[0] = "abandon"
	words0, _ := e.EnumerateAt(0, v)
	// WordList[1] = "ability"
	words1, _ := e.EnumerateAt(1, v)

	if words0 != nil && words1 != nil && words0[0] == words1[0] {
		t.Error("index 0 and index 1 should map to different words")
	}
	if words0 != nil && words0[0] != mnemonic.WordList[0] {
		t.Errorf("index 0 should map to %q, got %q", mnemonic.WordList[0], words0[0])
	}
	if words1 != nil && words1[0] != mnemonic.WordList[1] {
		t.Errorf("index 1 should map to %q, got %q", mnemonic.WordList[1], words1[0])
	}
}
