package protocol

import (
	"bytes"
	"testing"
)

// ---- TaskTemplate ----

func TestTaskTemplateEncodeDecode(t *testing.T) {
	original := &TaskTemplate{
		JobID:      42,
		UnknownPos: []int{0, 5, 11},
		Words: []string{
			"", "able", "above", "absent", "absorb", "", "abstract", "absurd", "abuse", "access", "accident", "",
		},
	}

	encoded := original.Encode()
	decoded, err := DecodeTemplate(encoded)
	if err != nil {
		t.Fatalf("DecodeTemplate failed: %v", err)
	}

	if decoded.JobID != original.JobID {
		t.Errorf("JobID = %d, want %d", decoded.JobID, original.JobID)
	}
	if len(decoded.UnknownPos) != len(original.UnknownPos) {
		t.Fatalf("UnknownPos length = %d, want %d", len(decoded.UnknownPos), len(original.UnknownPos))
	}
	for i, pos := range original.UnknownPos {
		if decoded.UnknownPos[i] != pos {
			t.Errorf("UnknownPos[%d] = %d, want %d", i, decoded.UnknownPos[i], pos)
		}
	}
	if len(decoded.Words) != 12 {
		t.Fatalf("Words length = %d, want 12", len(decoded.Words))
	}
	for i, word := range original.Words {
		if decoded.Words[i] != word {
			t.Errorf("Words[%d] = %q, want %q", i, decoded.Words[i], word)
		}
	}
}

func TestTaskTemplateEncodeDecodeNoUnknown(t *testing.T) {
	original := &TaskTemplate{
		JobID:      1,
		UnknownPos: []int{},
		Words:      []string{"abandon", "ability", "able", "about", "above", "absent", "absorb", "abstract", "absurd", "abuse", "access", "accident"},
	}

	encoded := original.Encode()
	decoded, err := DecodeTemplate(encoded)
	if err != nil {
		t.Fatalf("DecodeTemplate failed: %v", err)
	}
	if len(decoded.UnknownPos) != 0 {
		t.Errorf("UnknownPos = %v, want []", decoded.UnknownPos)
	}
	for i, word := range original.Words {
		if decoded.Words[i] != word {
			t.Errorf("Words[%d] = %q, want %q", i, decoded.Words[i], word)
		}
	}
}

func TestDecodeTemplateInvalidData(t *testing.T) {
	_, err := DecodeTemplate([]byte{0x00})
	if err == nil {
		t.Error("DecodeTemplate with too-short data should return error")
	}
}

// ---- CompactTask ----

func TestCompactTaskEncodeLength(t *testing.T) {
	task := &CompactTask{
		TaskID:   1,
		JobID:    2,
		StartIdx: 0,
		EndIdx:   10000,
	}
	encoded := task.Encode()
	if len(encoded) != 32 {
		t.Errorf("CompactTask.Encode() length = %d, want 32", len(encoded))
	}
}

func TestCompactTaskEncodeDecode(t *testing.T) {
	tests := []struct {
		name string
		task *CompactTask
	}{
		{
			name: "zero values",
			task: &CompactTask{0, 0, 0, 0},
		},
		{
			name: "positive values",
			task: &CompactTask{TaskID: 999, JobID: 42, StartIdx: 10000, EndIdx: 20000},
		},
		{
			name: "large values",
			task: &CompactTask{TaskID: 1<<62 - 1, JobID: 1<<32, StartIdx: 1 << 50, EndIdx: 1<<50 + 10000},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			encoded := tt.task.Encode()
			decoded, err := DecodeCompactTask(encoded)
			if err != nil {
				t.Fatalf("DecodeCompactTask failed: %v", err)
			}
			if *decoded != *tt.task {
				t.Errorf("decoded = %+v, want %+v", decoded, tt.task)
			}
		})
	}
}

func TestDecodeCompactTaskTooShort(t *testing.T) {
	_, err := DecodeCompactTask([]byte{0x01, 0x02})
	if err == nil {
		t.Error("DecodeCompactTask with too-short data should return error")
	}
}

// ---- CompactResult ----

func TestCompactResultEncodeDecodeNoMatches(t *testing.T) {
	original := &CompactResult{
		TaskID:  7,
		Matches: []MatchData{},
	}
	encoded := original.Encode()
	if len(encoded) != 12 {
		t.Errorf("encoded length with 0 matches = %d, want 12", len(encoded))
	}

	decoded, err := DecodeCompactResult(encoded)
	if err != nil {
		t.Fatalf("DecodeCompactResult failed: %v", err)
	}
	if decoded.TaskID != original.TaskID {
		t.Errorf("TaskID = %d, want %d", decoded.TaskID, original.TaskID)
	}
	if len(decoded.Matches) != 0 {
		t.Errorf("Matches length = %d, want 0", len(decoded.Matches))
	}
}

func TestCompactResultEncodeDecodeWithMatches(t *testing.T) {
	addr1 := []byte{0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13, 0x14}
	addr2 := []byte{0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff, 0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd}

	original := &CompactResult{
		TaskID: 99,
		Matches: []MatchData{
			{Index: 123, Address: addr1},
			{Index: 456, Address: addr2},
		},
	}

	encoded := original.Encode()
	// 12 header + 2*28 = 68
	if len(encoded) != 68 {
		t.Errorf("encoded length with 2 matches = %d, want 68", len(encoded))
	}

	decoded, err := DecodeCompactResult(encoded)
	if err != nil {
		t.Fatalf("DecodeCompactResult failed: %v", err)
	}
	if decoded.TaskID != original.TaskID {
		t.Errorf("TaskID = %d, want %d", decoded.TaskID, original.TaskID)
	}
	if len(decoded.Matches) != 2 {
		t.Fatalf("Matches length = %d, want 2", len(decoded.Matches))
	}
	if decoded.Matches[0].Index != 123 {
		t.Errorf("Matches[0].Index = %d, want 123", decoded.Matches[0].Index)
	}
	if !bytes.Equal(decoded.Matches[0].Address, addr1) {
		t.Errorf("Matches[0].Address = %x, want %x", decoded.Matches[0].Address, addr1)
	}
	if decoded.Matches[1].Index != 456 {
		t.Errorf("Matches[1].Index = %d, want 456", decoded.Matches[1].Index)
	}
	if !bytes.Equal(decoded.Matches[1].Address, addr2) {
		t.Errorf("Matches[1].Address = %x, want %x", decoded.Matches[1].Address, addr2)
	}
}

func TestDecodeCompactResultTooShort(t *testing.T) {
	_, err := DecodeCompactResult([]byte{0x01})
	if err == nil {
		t.Error("DecodeCompactResult with too-short data should return error")
	}
}

func TestDecodeCompactResultIncompleteMatch(t *testing.T) {
	// Build a result that claims 1 match but has no match data
	data := make([]byte, 12)
	data[11] = 1 // count = 1, but no match payload follows
	_, err := DecodeCompactResult(data)
	if err == nil {
		t.Error("DecodeCompactResult with incomplete match data should return error")
	}
}
