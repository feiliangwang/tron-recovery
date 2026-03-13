package mnemonic

import (
	"testing"
)

// The template with 11 "abandon" words and one unknown position.
// "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about"
// is a known-valid BIP39 mnemonic, so position 11 being "about" (index 3 in WordList) is valid.
var abandonTemplate = []string{
	"abandon", "abandon", "abandon", "abandon", "abandon", "abandon",
	"abandon", "abandon", "abandon", "abandon", "abandon", "?",
}

func TestEnumerateNoUnknownValid(t *testing.T) {
	words := []string{"abandon", "abandon", "abandon", "abandon", "abandon", "abandon", "abandon", "abandon", "abandon", "abandon", "abandon", "about"}
	e := NewEnumerator(words, 10)
	results := collectChan(e.Enumerate())
	if len(results) != 1 {
		t.Errorf("Enumerate with valid no-unknown mnemonic: got %d results, want 1", len(results))
	}
}

func TestEnumerateNoUnknownInvalid(t *testing.T) {
	// "abandon" * 12 fails BIP39 checksum
	words := []string{"abandon", "abandon", "abandon", "abandon", "abandon", "abandon", "abandon", "abandon", "abandon", "abandon", "abandon", "abandon"}
	e := NewEnumerator(words, 10)
	results := collectChan(e.Enumerate())
	if len(results) != 0 {
		t.Errorf("Enumerate with invalid no-unknown mnemonic: got %d results, want 0", len(results))
	}
}

func TestEnumerateOneUnknown(t *testing.T) {
	e := NewEnumerator(abandonTemplate, 100)
	results := collectChan(e.Enumerate())
	// There should be at least one valid result
	if len(results) == 0 {
		t.Error("Enumerate with 1 unknown should find at least one valid mnemonic")
	}
	// All results should be 12 words
	for _, r := range results {
		if len(r) != 12 {
			t.Errorf("result has %d words, want 12", len(r))
		}
	}
	// All results should be valid mnemonics
	v := NewValidator()
	for _, r := range results {
		if !v.Validate(r) {
			t.Errorf("Enumerate returned invalid mnemonic: %v", r)
		}
	}
}

func TestEnumerateKnownValidResult(t *testing.T) {
	// "abandon"*11 + "about" is a known-valid BIP39 mnemonic
	// WordList[3] == "about", so the result at index 3 should be found
	e := NewEnumerator(abandonTemplate, 100)
	results := collectChan(e.Enumerate())

	found := false
	for _, r := range results {
		if len(r) == 12 && r[11] == "about" {
			found = true
			break
		}
	}
	if !found {
		t.Error("Enumerate did not find the known-valid mnemonic with 'about' at position 11")
	}
}

func TestBatchEnumerate(t *testing.T) {
	e := NewEnumerator(abandonTemplate, 1) // batch size 1
	var batches [][][]string
	for batch := range e.BatchEnumerate() {
		batches = append(batches, batch)
	}
	// Each batch should have at most 1 entry
	for _, b := range batches {
		if len(b) > 1 {
			t.Errorf("batch size > 1 with batchSize=1: got %d", len(b))
		}
	}
	// Total entries across batches should match Enumerate results
	e2 := NewEnumerator(abandonTemplate, 100)
	allResults := collectChan(e2.Enumerate())

	total := 0
	for _, b := range batches {
		total += len(b)
	}
	if total != len(allResults) {
		t.Errorf("BatchEnumerate total = %d, want %d", total, len(allResults))
	}
}

func TestBatchEnumerateLargeBatch(t *testing.T) {
	e := NewEnumerator(abandonTemplate, 10000) // batch size larger than results
	var batches [][][]string
	for batch := range e.BatchEnumerate() {
		batches = append(batches, batch)
	}
	// Should produce at most 1 batch (if there are any results)
	if len(batches) > 1 {
		t.Errorf("with batch size > total results, expected 1 batch, got %d", len(batches))
	}
}

func collectChan(ch <-chan []string) [][]string {
	var results [][]string
	for r := range ch {
		results = append(results, r)
	}
	return results
}
