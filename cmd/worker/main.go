package main

import (
	"bytes"
	"context"
	"encoding/hex"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"
	"runtime"
	"time"

	bip39 "github.com/tyler-smith/go-bip39"

	"boon/internal/bloom"
	"boon/internal/compute"
	"boon/internal/worker"
)

var (
	schedulerURL = flag.String("scheduler", "http://localhost:8080", "调度服务器地址")
	workerID     = flag.String("id", "", "Worker ID（留空自动生成）")
	workers      = flag.Int("workers", runtime.NumCPU(), "并发计算线程数")
	bloomFile    = flag.String("bloom", "account.bin.bloom", "Bloom过滤器文件（本地加载）")
	useGPU       = flag.Bool("gpu", false, "使用GPU加速（需要CUDA构建）")
	benchN       = flag.Int("bench", 0, "测速模式：随机生成N个助记词测算计算速度（0=禁用）")
	verifyN      = flag.Int("verify", 0, "验证模式：随机生成N个助记词对比GPU与CPU结果（0=禁用）")
)
func main() {
	flag.Parse()

	// 独立模式：测速
	if *benchN > 0 {
		runBench(*benchN, *useGPU, *workers)
		return
	}

	// 独立模式：验证
	if *verifyN > 0 {
		runVerify(*verifyN, *workers)
		return
	}

	// 正常 Worker 模式
	id := *workerID
	if id == "" {
		hostname, _ := os.Hostname()
		id = fmt.Sprintf("%s-%d", hostname, os.Getpid())
	}

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
	log.Printf("  GPU加速:       %s", boolStr(*useGPU, "启用", "禁用"))
	log.Printf("========================================")

	var seedComp compute.SeedComputer
	if *useGPU {
		gpu, err := compute.NewGPUComputer()
		if err != nil {
			log.Printf("GPU初始化失败，回退到CPU: %v", err)
			seedComp = compute.NewCPUComputer()
		} else {
			log.Printf("GPU计算器初始化成功")
			seedComp = gpu
		}
	} else {
		seedComp = compute.NewCPUComputer()
	}

	client := worker.NewCompactClient(*schedulerURL)
	w := worker.NewCompactWorkerWithComputer(id, client, *workers, seedComp)
	w.SetBloomFilter(bloomFilter)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		sig := <-sigCh
		log.Printf("收到信号: %v，正在关闭...", sig)
		cancel()
	}()

	startTime := time.Now()
	w.Start(ctx)
	<-ctx.Done()
	w.Stop()

	elapsed := time.Since(startTime)
	log.Printf("========================================")
	log.Printf("  Worker 已停止，运行时间: %v", elapsed)
	log.Printf("========================================")
}

// ================================================================
// 测速模式
// ================================================================

func runBench(total int, gpu bool, cpuWorkers int) {
	// 生成助记词
	fmt.Printf("生成 %d 个随机助记词...\n", total)
	mnemonics, err := genMnemonics(total)
	if err != nil {
		log.Fatalf("生成助记词失败: %v", err)
	}

	// 选择计算引擎
	var comp compute.SeedComputer
	compName := "CPU"
	if gpu {
		g, err := compute.NewGPUComputer()
		if err != nil {
			log.Fatalf("GPU初始化失败: %v\n提示: 需要 CUDA 构建且有可用 GPU", err)
		}
		comp = g
		compName = "GPU"
	} else {
		comp = compute.NewCPUComputer()
		compName = fmt.Sprintf("CPU(单线程)")
	}

	fmt.Printf("\n计算引擎: %s\n", compName)
	fmt.Printf("助记词总数: %d\n\n", total)

	batchSizes := []int{1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048}

	// 表头
	fmt.Printf("%-10s  %-12s  %-10s  %-14s\n", "批次大小", "已计算", "耗时", "速度(个/s)")
	fmt.Printf("%-10s  %-12s  %-10s  %-14s\n",
		"----------", "------------", "----------", "--------------")

	// 热身（第一次 CUDA kernel launch 较慢）
	warmupN := 32
	if warmupN > total {
		warmupN = total
	}
	comp.Compute(mnemonics[:warmupN])

	for _, bs := range batchSizes {
		if bs > total {
			break
		}
		computed := 0
		start := time.Now()
		for i := 0; i+bs <= total; i += bs {
			comp.Compute(mnemonics[i : i+bs])
			computed += bs
		}
		elapsed := time.Since(start)
		speed := float64(computed) / elapsed.Seconds()
		fmt.Printf("%-10d  %-12d  %-10s  %-.0f\n",
			bs, computed, fmtDuration(elapsed), speed)
	}

	fmt.Println()
}

// ================================================================
// 验证模式
// ================================================================

func runVerify(total int, cpuWorkers int) {
	gpuComp, err := compute.NewGPUComputer()
	if err != nil {
		log.Fatalf("GPU初始化失败: %v\n提示: 需要 CUDA 构建且有可用 GPU", err)
	}
	cpuComp := compute.NewCPUComputer()

	fmt.Printf("生成 %d 个随机助记词...\n", total)
	mnemonics, err := genMnemonics(total)
	if err != nil {
		log.Fatalf("生成助记词失败: %v", err)
	}

	// 用不同批次大小分段验证，确保边界情况也测到
	batchSizes := buildVerifyBatches(total)

	fmt.Printf("\n验证 GPU(compact) vs CPU(compact) 地址一致性\n")
	fmt.Printf("助记词总数: %d\n\n", total)

	totalChecked := 0
	totalFail := 0
	offset := 0

	fmt.Printf("%-10s  %-8s  %-8s  %-8s  %s\n", "批次大小", "已校验", "通过", "失败", "状态")
	fmt.Printf("%-10s  %-8s  %-8s  %-8s  %s\n",
		"----------", "--------", "--------", "--------", "------")

	for _, bs := range batchSizes {
		end := offset + bs
		if end > total {
			end = total
		}
		batch := mnemonics[offset:end]
		n := len(batch)
		if n == 0 {
			break
		}

		gpuAddrs := gpuComp.Compute(batch)
		cpuAddrs := cpuComp.Compute(batch)

		fail := 0
		for i := range batch {
			if !bytes.Equal(gpuAddrs[i], cpuAddrs[i]) {
				fail++
				fmt.Printf("  MISMATCH [%d] mnemonic: %q\n", offset+i, batch[i])
				fmt.Printf("    GPU: %s\n", hex.EncodeToString(gpuAddrs[i]))
				fmt.Printf("    CPU: %s\n", hex.EncodeToString(cpuAddrs[i]))
			}
		}

		totalChecked += n
		totalFail += fail
		offset = end

		status := "OK"
		if fail > 0 {
			status = "FAIL"
		}
		fmt.Printf("%-10d  %-8d  %-8d  %-8d  %s\n",
			bs, totalChecked, totalChecked-totalFail, totalFail, status)

		if offset >= total {
			break
		}
	}

	fmt.Println()
	if totalFail == 0 {
		fmt.Printf("✓ 全部通过  %d/%d 个地址与CPU一致\n", totalChecked, totalChecked)
	} else {
		fmt.Printf("✗ 验证失败  %d/%d 个不一致\n", totalFail, totalChecked)
		os.Exit(1)
	}
}

// buildVerifyBatches 返回覆盖 total 个元素的批次序列：
// 先用小批次（边界测试），再用大批次。
func buildVerifyBatches(total int) []int {
	sizes := []int{1, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}
	var result []int
	remaining := total
	for _, s := range sizes {
		if remaining <= 0 {
			break
		}
		if s > remaining {
			s = remaining
		}
		result = append(result, s)
		remaining -= s
	}
	// 剩余部分用 512 块
	for remaining > 0 {
		s := 512
		if s > remaining {
			s = remaining
		}
		result = append(result, s)
		remaining -= s
	}
	return result
}

// ================================================================
// 工具函数
// ================================================================

func genMnemonics(n int) ([]string, error) {
	out := make([]string, n)
	for i := range out {
		entropy, err := bip39.NewEntropy(128) // 12 words
		if err != nil {
			return nil, err
		}
		mn, err := bip39.NewMnemonic(entropy)
		if err != nil {
			return nil, err
		}
		out[i] = mn
	}
	return out, nil
}

func fmtDuration(d time.Duration) string {
	if d < time.Millisecond {
		return fmt.Sprintf("%dµs", d.Microseconds())
	}
	if d < time.Second {
		return fmt.Sprintf("%dms", d.Milliseconds())
	}
	return fmt.Sprintf("%.2fs", d.Seconds())
}

func boolStr(cond bool, trueStr, falseStr string) string {
	if cond {
		return trueStr
	}
	return falseStr
}
