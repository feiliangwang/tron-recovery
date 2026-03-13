package compute

// SeedComputer 种子计算器接口
// 输入：一批助记词
// 输出：一批20字节的地址（公钥Keccak256前20字节）
type SeedComputer interface {
	// Compute 计算助记词对应的TRON地址
	// mnemonics: 一批助记词，每个助记词是12个词的切片
	// 返回: 一批20字节的地址
	Compute(mnemonics []string) [][]byte

	// Close 关闭计算器，释放资源
	Close() error
}
