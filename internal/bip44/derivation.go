package bip44

import (
	"crypto/sha256"

	"github.com/btcsuite/btcd/btcutil/base58"
)

//
//const (
//	// Purpose BIP44 purpose
//	Purpose = 44
//	// CoinTypeTRON TRON coin type
//	CoinTypeTRON = 195
//)
//
//// Deriver 密钥派生器
//type Deriver struct {
//	seed []byte
//}
//
//// NewDeriverFromSeed 从种子创建派生器
//func NewDeriverFromSeed(seed []byte) *Deriver {
//	return &Deriver{seed: seed}
//}
//
//// NewDeriverFromMnemonic 从助记词创建派生器
//func NewDeriverFromMnemonic(mnemonic string) (*Deriver, error) {
//	seed := bip39.NewSeed(mnemonic, "")
//	return NewDeriverFromSeed(seed), nil
//}
//
//// DeriveMasterKey 派生主密钥
//func (d *Deriver) DeriveMasterKey() (*hdkeychain.ExtendedKey, error) {
//	return hdkeychain.NewMaster(d.seed, &chaincfg.MainNetParams)
//}
//
//// DerivePath 派生指定路径的密钥
//// 路径格式: m/44'/195'/0'/0/0
//func (d *Deriver) DerivePath(path []uint32) (*hdkeychain.ExtendedKey, error) {
//	masterKey, err := d.DeriveMasterKey()
//	if err != nil {
//		return nil, fmt.Errorf("failed to derive master key: %w", err)
//	}
//
//	currentKey := masterKey
//	for _, childNum := range path {
//		childKey, err := currentKey.Derive(childNum)
//		if err != nil {
//			return nil, fmt.Errorf("failed to derive child key: %w", err)
//		}
//		currentKey = childKey
//	}
//
//	return currentKey, nil
//}
//
//// DeriveTRON 派生TRON路径 m/44'/195'/0'/0/0
//func (d *Deriver) DeriveTRON() (*hdkeychain.ExtendedKey, error) {
//	// BIP44 路径: m / purpose' / coin_type' / account' / change / address_index
//	// TRON: m / 44' / 195' / 0' / 0 / 0
//	path := []uint32{
//		hdkeychain.HardenedKeyStart + Purpose,      // 44'
//		hdkeychain.HardenedKeyStart + CoinTypeTRON, // 195'
//		hdkeychain.HardenedKeyStart + 0,            // 0'
//		0,                                          // 0 (external chain)
//		0,                                          // 0 (first address)
//	}
//	return d.DerivePath(path)
//}
//
//// GetPrivateKey 获取私钥
//func GetPrivateKey(key *hdkeychain.ExtendedKey) (*btcec.PrivateKey, error) {
//	return key.ECPrivKey()
//}
//
//// GetPublicKey 获取公钥
//func GetPublicKey(key *hdkeychain.ExtendedKey) (*btcec.PublicKey, error) {
//	privKey, err := key.ECPrivKey()
//	if err != nil {
//		return nil, err
//	}
//	return privKey.PubKey(), nil
//}
//
//// GetPublicKeyBytes 获取公钥字节（未压缩格式，去掉前缀04）
//func GetPublicKeyBytes(key *hdkeychain.ExtendedKey) ([]byte, error) {
//	pubKey, err := GetPublicKey(key)
//	if err != nil {
//		return nil, err
//	}
//
//	// 获取未压缩公钥序列化（65字节：04 + X + Y）
//	serialized := pubKey.SerializeUncompressed()
//
//	// 去掉04前缀，返回64字节
//	if len(serialized) == 65 && serialized[0] == 0x04 {
//		return serialized[1:], nil
//	}
//
//	return serialized, nil
//}

func GetTronAddress(addr20 []byte) string {
	// 8. TRON 前缀 0x41
	addr := append([]byte{0x41}, addr20...)
	// 9. Base58Check
	c1 := sha256.Sum256(addr)
	c2 := sha256.Sum256(c1[:])
	checksum := c2[:4]
	address := base58.Encode(append(addr, checksum...))
	return address
}
