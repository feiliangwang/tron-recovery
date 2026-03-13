package bip44

import (
	"testing"
)

func TestGetTronAddress(t *testing.T) {
	// 已知的20字节地址
	addr20 := []byte{
		0x29, 0x6e, 0x1e, 0x73, 0x48, 0x97, 0xfc, 0x64, 0xd4, 0x0b,
		0x17, 0xda, 0xbd, 0x0a, 0xb4, 0xa8, 0x12, 0x74, 0x85, 0x42,
	}
	got := GetTronAddress(addr20)
	want := "TDkGd84eSuvFcUXqnvNY8UDmAj1ZE6umZJ"
	if got != want {
		t.Errorf("GetTronAddress = %q, want %q", got, want)
	}
}

func TestGetTronAddressLength(t *testing.T) {
	addr20 := make([]byte, 20)
	addr := GetTronAddress(addr20)
	// TRON Base58Check 地址固定34字节
	if len(addr) != 34 {
		t.Errorf("TRON address length = %d, want 34", len(addr))
	}
}

func TestGetTronAddressDifferentInputs(t *testing.T) {
	addr1 := make([]byte, 20)
	addr2 := make([]byte, 20)
	addr2[0] = 0x01

	t1 := GetTronAddress(addr1)
	t2 := GetTronAddress(addr2)
	if t1 == t2 {
		t.Error("different 20-byte inputs should produce different TRON addresses")
	}
}

func TestGetTronAddressDeterministic(t *testing.T) {
	addr20 := make([]byte, 20)
	for i := range addr20 {
		addr20[i] = byte(i)
	}
	a1 := GetTronAddress(addr20)
	a2 := GetTronAddress(addr20)
	if a1 != a2 {
		t.Error("GetTronAddress should be deterministic")
	}
}
