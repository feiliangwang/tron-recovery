package compute

import (
	"github.com/btcsuite/btcd/btcutil/hdkeychain"
	"github.com/btcsuite/btcd/chaincfg"
	"github.com/tyler-smith/go-bip39"
	"golang.org/x/crypto/sha3"
)

// CPUComputer CPU计算器
type CPUComputer struct {
}

// NewCPUComputer 创建CPU计算器
func NewCPUComputer() *CPUComputer {
	return &CPUComputer{}
}

// Compute 计算助记词对应的TRON地址
func (c *CPUComputer) Compute(mnemonics []string) [][]byte {
	addresses := make([][]byte, len(mnemonics))
	for i, mnemonic := range mnemonics {
		addresses[i] = c.computeOne(mnemonic)
	}
	return addresses
}

// computeOne 计算单个助记词对应的TRON地址
func (c *CPUComputer) computeOne(mnemonic string) []byte {
	seed := bip39.NewSeed(mnemonic, "")
	// 3. BIP44 派生 TRON 路径 m/44'/195'/0'/0/0
	// 3. m/44'/195'/0'/0'/0'
	masterKey, err := hdkeychain.NewMaster(seed, &chaincfg.MainNetParams)
	if err != nil {
		panic(err)
	}
	purpose, err := masterKey.Derive(hdkeychain.HardenedKeyStart + 44)
	if err != nil {
		panic(err)
	}
	coinType, err := purpose.Derive(hdkeychain.HardenedKeyStart + 195)
	if err != nil {
		panic(err)
	}
	account, err := coinType.Derive(hdkeychain.HardenedKeyStart + 0)
	if err != nil {
		panic(err)
	}
	change, err := account.Derive(0)
	if err != nil {
		panic(err)
	}
	addrKey, err := change.Derive(0)
	if err != nil {
		panic(err)
	}
	// 4. private key
	privKey, err := addrKey.ECPrivKey()
	if err != nil {
		panic(err)
	}
	pubBytes := privKey.PubKey().SerializeUncompressed()[1:] // 去掉 0x04
	hash := sha3.NewLegacyKeccak256()
	hash.Write(pubBytes)
	sum := hash.Sum(nil)
	return sum[len(sum)-20:]
}

// Close 关闭CPU计算器
func (c *CPUComputer) Close() error {
	return nil
}
