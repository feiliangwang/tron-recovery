package main

import (
	"bufio"
	"encoding/hex"
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
		fmt.Println("用法: bloomtool -input addresses.txt -output bloom.gob")
		flag.PrintDefaults()
		os.Exit(1)
	}

	fmt.Printf("读取地址文件: %s\n", *inputFile)

	// 读取所有地址
	file, err := os.Open(*inputFile)
	if err != nil {
		fmt.Fprintf(os.Stderr, "打开文件失败: %v\n", err)
		os.Exit(1)
	}
	defer file.Close()

	var addresses [][]byte
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		if len(line) == 0 {
			continue
		}

		addr, err := hex.DecodeString(line)
		if err != nil {
			fmt.Fprintf(os.Stderr, "跳过无效地址: %s (err: %v)\n", line, err)
			continue
		}

		if len(addr) == 20 {
			addresses = append(addresses, addr)
		}
	}

	if len(addresses) == 0 {
		fmt.Fprintln(os.Stderr, "没有有效的地址")
		os.Exit(1)
	}

	fmt.Printf("读取到 %d 个有效地址\n", len(addresses))

	// 创建Bloom过滤器
	start := time.Now()
	filter := bloom.NewFilter(uint(len(addresses)*2), *falseRate)

	for _, addr := range addresses {
		filter.Add(addr)
	}

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
