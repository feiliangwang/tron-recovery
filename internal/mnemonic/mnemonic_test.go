package mnemonic

import (
	"reflect"
	"testing"
)

const validMnemonic = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about"

func TestWordList(t *testing.T) {
	if len(WordList) != 2048 {
		t.Errorf("WordList length = %d, want 2048", len(WordList))
	}
	if WordCount != 2048 {
		t.Errorf("WordCount = %d, want 2048", WordCount)
	}
	if WordList[0] != "abandon" {
		t.Errorf("WordList[0] = %q, want %q", WordList[0], "abandon")
	}
}

func TestValidatorValidate(t *testing.T) {
	v := NewValidator()

	tests := []struct {
		name  string
		words []string
		want  bool
	}{
		{
			name:  "valid 12-word mnemonic",
			words: []string{"abandon", "abandon", "abandon", "abandon", "abandon", "abandon", "abandon", "abandon", "abandon", "abandon", "abandon", "about"},
			want:  true,
		},
		{
			name:  "wrong word count",
			words: []string{"abandon", "abandon", "abandon"},
			want:  false,
		},
		{
			name:  "invalid word",
			words: []string{"notaword", "abandon", "abandon", "abandon", "abandon", "abandon", "abandon", "abandon", "abandon", "abandon", "abandon", "about"},
			want:  false,
		},
		{
			name:  "bad checksum",
			words: []string{"abandon", "abandon", "abandon", "abandon", "abandon", "abandon", "abandon", "abandon", "abandon", "abandon", "abandon", "abandon"},
			want:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := v.Validate(tt.words)
			if got != tt.want {
				t.Errorf("Validate(%v) = %v, want %v", tt.words, got, tt.want)
			}
		})
	}
}

func TestValidatorValidateMnemonic(t *testing.T) {
	v := NewValidator()

	if !v.ValidateMnemonic(validMnemonic) {
		t.Errorf("ValidateMnemonic(%q) = false, want true", validMnemonic)
	}

	if v.ValidateMnemonic("invalid mnemonic string") {
		t.Error("ValidateMnemonic(invalid) = true, want false")
	}
}

func TestGetUnknownIndices(t *testing.T) {
	tests := []struct {
		name  string
		words []string
		want  []int
	}{
		{
			name:  "no unknowns",
			words: []string{"word1", "word2", "word3"},
			want:  []int{},
		},
		{
			name:  "one unknown at start",
			words: []string{"?", "word2", "word3"},
			want:  []int{0},
		},
		{
			name:  "one unknown at end",
			words: []string{"word1", "word2", "?"},
			want:  []int{2},
		},
		{
			name:  "multiple unknowns",
			words: []string{"?", "word2", "?", "word4", "?"},
			want:  []int{0, 2, 4},
		},
		{
			name:  "all unknowns",
			words: []string{"?", "?", "?"},
			want:  []int{0, 1, 2},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := GetUnknownIndices(tt.words)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("GetUnknownIndices(%v) = %v, want %v", tt.words, got, tt.want)
			}
		})
	}
}

func TestReplaceWords(t *testing.T) {
	tests := []struct {
		name         string
		words        []string
		indices      []int
		replacements []string
		want         []string
	}{
		{
			name:         "replace single word",
			words:        []string{"a", "b", "c"},
			indices:      []int{1},
			replacements: []string{"X"},
			want:         []string{"a", "X", "c"},
		},
		{
			name:         "replace multiple words",
			words:        []string{"a", "b", "c", "d"},
			indices:      []int{0, 3},
			replacements: []string{"X", "Y"},
			want:         []string{"X", "b", "c", "Y"},
		},
		{
			name:         "fewer replacements than indices",
			words:        []string{"a", "b", "c"},
			indices:      []int{0, 1, 2},
			replacements: []string{"X"},
			want:         []string{"X", "b", "c"},
		},
		{
			name:         "no replacements",
			words:        []string{"a", "b", "c"},
			indices:      []int{},
			replacements: []string{},
			want:         []string{"a", "b", "c"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ReplaceWords(tt.words, tt.indices, tt.replacements)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("ReplaceWords() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestReplaceWordsDoesNotMutateOriginal(t *testing.T) {
	original := []string{"a", "b", "c"}
	orig := make([]string, len(original))
	copy(orig, original)

	ReplaceWords(original, []int{0}, []string{"X"})

	if !reflect.DeepEqual(original, orig) {
		t.Error("ReplaceWords mutated the original slice")
	}
}
