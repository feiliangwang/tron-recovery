package worker

import (
	"boon/internal/mnemonic"
	"strings"
)

// TaskTemplate 任务模板
type TaskTemplate struct {
	JobID      int64
	Words      []string // 12个词，空字符串=需要枚举
	UnknownPos []int    // 需要枚举的位置索引
}

// LocalEnumerator 本地枚举器
type LocalEnumerator struct {
	template     *TaskTemplate
	wordList     []string
	wordCount    int64
}

// NewLocalEnumerator 创建本地枚举器
func NewLocalEnumerator(template *TaskTemplate) *LocalEnumerator {
	// 计算未知位置
	if len(template.UnknownPos) == 0 {
		unknownPos := make([]int, 0)
		for i, word := range template.Words {
			if word == "" {
				unknownPos = append(unknownPos, i)
			}
		}
		template.UnknownPos = unknownPos
	}

	return &LocalEnumerator{
		template:  template,
		wordList:  mnemonic.WordList,
		wordCount: int64(len(mnemonic.WordList)),
	}
}

// EnumerateAt 枚举指定索引位置的助记词
// 实现 compute.Enumerator 接口
func (e *LocalEnumerator) EnumerateAt(idx int64, validator *mnemonic.Validator) ([]string, bool) {
	words := make([]string, len(e.template.Words))
	copy(words, e.template.Words)

	// 将索引转换为各位置的词索引
	remaining := idx
	for _, pos := range e.template.UnknownPos {
		wordIdx := remaining % e.wordCount
		remaining /= e.wordCount
		words[pos] = e.wordList[wordIdx]
	}

	if !validator.Validate(words) {
		return nil, false
	}

	return words, true
}

// GetMnemonic 获取助记词字符串
func (e *LocalEnumerator) GetMnemonic(words []string) string {
	return strings.Join(words, " ")
}

// GetWordCount 获取词表大小
func (e *LocalEnumerator) GetWordCount() int64 {
	return e.wordCount
}
