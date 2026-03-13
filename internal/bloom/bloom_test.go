package bloom

import (
	"os"
	"path/filepath"
	"testing"
)

func TestNewFilter(t *testing.T) {
	f := NewFilter(1000, 0.01)
	if f == nil {
		t.Fatal("NewFilter returned nil")
	}
	if f.filter == nil {
		t.Fatal("NewFilter: inner filter is nil")
	}
}

func TestAddAndContains(t *testing.T) {
	f := NewFilter(100, 0.001)

	items := [][]byte{
		{0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13, 0x14},
		{0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff, 0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd},
		[]byte("testaddress1234567890"),
	}

	for _, item := range items {
		f.Add(item)
	}

	// No false negatives allowed
	for _, item := range items {
		if !f.Contains(item) {
			t.Errorf("Contains returned false for added item %x", item)
		}
	}
}

func TestContainsReturnsFalseForUnaddedItem(t *testing.T) {
	f := NewFilter(1000, 0.0001)

	f.Add([]byte("item1"))
	f.Add([]byte("item2"))

	// With very low false positive rate, this should not be in the filter
	// (there's a tiny chance it could be a false positive, but with 0.0001 rate it's very unlikely)
	if f.Contains([]byte("definitely_not_added_12345678901234567890")) {
		// This is a probabilistic test - log but don't fail
		t.Log("false positive detected (extremely rare but valid)")
	}
}

func TestClear(t *testing.T) {
	f := NewFilter(100, 0.01)

	items := [][]byte{
		[]byte("item1"),
		[]byte("item2"),
		[]byte("item3"),
	}
	for _, item := range items {
		f.Add(item)
	}

	f.Clear()

	for _, item := range items {
		if f.Contains(item) {
			t.Errorf("After Clear, Contains still returns true for %q", item)
		}
	}
}

func TestTestAndAdd(t *testing.T) {
	f := NewFilter(100, 0.001)
	item := []byte("test_item")

	// First call: item not yet added, should return false
	if f.TestAndAdd(item) {
		t.Error("TestAndAdd on new item should return false")
	}

	// Second call: item was added by previous call, should return true
	if !f.TestAndAdd(item) {
		t.Error("TestAndAdd on existing item should return true")
	}
}

func TestSaveAndLoadFromFile(t *testing.T) {
	dir := t.TempDir()
	filename := filepath.Join(dir, "test.bloom")

	// Create and populate filter
	f := NewFilter(100, 0.001)
	items := [][]byte{
		{0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff, 0x00, 0x01, 0x02, 0x03, 0x04},
		{0xde, 0xad, 0xbe, 0xef, 0xca, 0xfe, 0xba, 0xbe, 0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb},
	}
	for _, item := range items {
		f.Add(item)
	}

	// Save to file
	if err := f.SaveToFile(filename); err != nil {
		t.Fatalf("SaveToFile failed: %v", err)
	}

	// Verify file exists
	if _, err := os.Stat(filename); err != nil {
		t.Fatalf("file not created: %v", err)
	}

	// Load from file
	loaded, err := LoadFromFile(filename)
	if err != nil {
		t.Fatalf("LoadFromFile failed: %v", err)
	}

	// Loaded filter must contain all previously added items (no false negatives)
	for _, item := range items {
		if !loaded.Contains(item) {
			t.Errorf("Loaded filter does not contain item %x", item)
		}
	}
}

func TestLoadFromFileNotFound(t *testing.T) {
	_, err := LoadFromFile("/nonexistent/path/filter.bloom")
	if err == nil {
		t.Error("LoadFromFile on missing file should return error")
	}
}

func TestSaveToFileInvalidPath(t *testing.T) {
	f := NewFilter(10, 0.01)
	err := f.SaveToFile("/nonexistent/path/filter.bloom")
	if err == nil {
		t.Error("SaveToFile to invalid path should return error")
	}
}
