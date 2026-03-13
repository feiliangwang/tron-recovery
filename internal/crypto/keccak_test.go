package crypto

import (
	"bytes"
	"encoding/hex"
	"testing"
)

func TestKeccak256KnownVectors(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  string
	}{
		{
			name:  "empty input",
			input: "",
			want:  "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470",
		},
		{
			name:  "hello",
			input: "hello",
			want:  "1c8aff950685c2ed4bc3174f3472287b56d9517b9c948127319a09a7a36deac8",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Keccak256([]byte(tt.input))
			if hex.EncodeToString(got) != tt.want {
				t.Errorf("Keccak256(%q) = %x, want %s", tt.input, got, tt.want)
			}
		})
	}
}

func TestKeccak256OutputLength(t *testing.T) {
	inputs := [][]byte{
		{},
		[]byte("a"),
		[]byte("hello world"),
		make([]byte, 1000),
	}
	for _, input := range inputs {
		got := Keccak256(input)
		if len(got) != 32 {
			t.Errorf("Keccak256 output length = %d, want 32", len(got))
		}
	}
}

func TestKeccak256Deterministic(t *testing.T) {
	data := []byte("test determinism")
	first := Keccak256(data)
	second := Keccak256(data)
	if !bytes.Equal(first, second) {
		t.Error("Keccak256 is not deterministic")
	}
}

func TestKeccak256DifferentInputs(t *testing.T) {
	a := Keccak256([]byte("input a"))
	b := Keccak256([]byte("input b"))
	if bytes.Equal(a, b) {
		t.Error("Keccak256 of different inputs should not be equal")
	}
}

func TestKeccak256Hash(t *testing.T) {
	data := []byte("test data")
	result := Keccak256Hash(data)
	if len(result) != 20 {
		t.Errorf("Keccak256Hash output length = %d, want 20", len(result))
	}

	// must equal the first 20 bytes of Keccak256
	full := Keccak256(data)
	if !bytes.Equal(result, full[:20]) {
		t.Errorf("Keccak256Hash mismatch: got %x, want %x", result, full[:20])
	}
}

func TestKeccak256HashDeterministic(t *testing.T) {
	data := []byte("test")
	first := Keccak256Hash(data)
	second := Keccak256Hash(data)
	if !bytes.Equal(first, second) {
		t.Error("Keccak256Hash is not deterministic")
	}
}
