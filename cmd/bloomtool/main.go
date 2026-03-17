package main

import (
	"boon/internal/account"
	"flag"
	"fmt"
	"os"
	"time"

	"boon/internal/bloom"
)

var (
	inputFile  = flag.String("input", "", "输入文件（每行一个hex地址）")
	outputFile = flag.String("output", "bloom.gob", "输出文件（gob格式）")
	falseRate  = flag.Float64("rate", 0.001, "误判率")
)

func main() {
	flag.Parse()

	if *inputFile == "" {
		fmt.Println("用法: bloomtool -input leveldb -output bloom.gob")
		flag.PrintDefaults()
		os.Exit(1)
	}

	fmt.Printf("读取地址文件: %s\n", *inputFile)

	// 读取所有地址
	db, err := account.NewAccountDb(*inputFile)
	if err != nil {
		fmt.Fprintf(os.Stderr, "打开文件失败: %v\n", err)
		os.Exit(1)
	}
	start := time.Now()
	filter := bloom.NewFilter(uint(db.Count()*2), *falseRate)
	db.IteratorAccount(func(addr20 []byte) {
		filter.Add(addr20)
	})
	fmt.Printf("Bloom过滤器创建完成，耗时: %v\n", time.Since(start))

	// 保存到文件
	if err := filter.SaveToFile(*outputFile); err != nil {
		fmt.Fprintf(os.Stderr, "保存失败: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Bloom过滤器已保存到: %s\n", *outputFile)

	// 验证
	stat, _ := os.Stat(*outputFile)
	fmt.Printf("文件大小: %d bytes (%.2f MB)\n", stat.Size(), float64(stat.Size())/1024/1024)
}
