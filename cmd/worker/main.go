package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"runtime"
	"syscall"
	"time"

	"boon/internal/bloom"
	"boon/internal/worker"
)

var (
	schedulerURL = flag.String("scheduler", "http://localhost:8080", "调度服务器地址")
	workerID     = flag.String("id", "", "Worker ID（留空自动生成）")
	workers      = flag.Int("workers", runtime.NumCPU(), "并发计算线程数")
	bloomFile    = flag.String("bloom", "account.bin.bloom", "Bloom过滤器文件（本地加载）")
)

func main() {
	flag.Parse()

	// 生成Worker ID
	id := *workerID
	if id == "" {
		hostname, _ := os.Hostname()
		id = fmt.Sprintf("%s-%d", hostname, os.Getpid())
	}

	// 加载Bloom过滤器
	var bloomFilter *bloom.Filter
	if *bloomFile != "" {
		log.Printf("加载Bloom过滤器: %s", *bloomFile)
		var err error
		bloomFilter, err = bloom.LoadFromFile(*bloomFile)
		if err != nil {
			log.Fatalf("加载Bloom过滤器失败: %v", err)
		}
		log.Println("Bloom过滤器加载完成")
	}

	log.Printf("========================================")
	log.Printf("  Boon Worker v2 (紧凑协议)")
	log.Printf("========================================")
	log.Printf("  ID:           %s", id)
	log.Printf("  调度服务器:    %s", *schedulerURL)
	log.Printf("  计算线程:      %d", *workers)
	log.Printf("  Bloom过滤:     %s", boolStr(bloomFilter != nil, "已加载", "未加载"))
	log.Printf("========================================")

	// 创建紧凑协议客户端
	client := worker.NewCompactClient(*schedulerURL)

	// 创建紧凑Worker
	w := worker.NewCompactWorker(id, client, *workers)
	w.SetBloomFilter(bloomFilter)

	// 处理信号
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		sig := <-sigCh
		log.Printf("收到信号: %v，正在关闭...", sig)
		cancel()
	}()

	// 启动Worker
	startTime := time.Now()
	w.Start(ctx)

	// 等待完成
	<-ctx.Done()
	w.Stop()

	// 打印最终统计
	elapsed := time.Since(startTime)
	log.Printf("========================================")
	log.Printf("  Worker 已停止")
	log.Printf("  运行时间: %v", elapsed)
	log.Printf("========================================")
}

func boolStr(cond bool, trueStr, falseStr string) string {
	if cond {
		return trueStr
	}
	return falseStr
}
