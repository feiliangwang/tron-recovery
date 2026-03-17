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
	"runtime"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	bip39 "github.com/tyler-smith/go-bip39"

	"boon/internal/bloom"
	"boon/internal/compute"
	mn "boon/internal/mnemonic"
	"boon/internal/protocol"
	"boon/internal/worker"
)

var (
	schedulerURL   = flag.String("scheduler", "http://192.168.3.250", "调度服务器地址")
	workerID       = flag.String("id", "", "Worker ID（留空自动生成）")
	workers        = flag.Int("workers", runtime.NumCPU(), "并发计算线程数（CPU模式）")
	bloomFile      = flag.String("bloom", "account.bin.bloom", "Bloom过滤器文件（本地加载）")
	useGPU         = flag.Bool("gpu", false, "使用全部可用GPU（等价于 -gpu-devices all）")
	gpuDevices     = flag.String("gpu-devices", "", "指定GPU设备列表，如 '0,1,2' 或 'all'（覆盖 -gpu 标志）")
	benchN         = flag.Int("bench", 0, "测速模式：随机生成N个助记词测算计算速度（0=禁用）")
	verifyN        = flag.Int("verify", 0, "验证模式：随机生成N个助记词对比GPU与CPU结果（0=禁用）")
	benchFullN     = flag.Int64("bench-full", 0, "全链路测速：模拟完整枚举→验证→计算流程，扫描N个索引（0=禁用）")
	bloomTestMnem  = flag.String("bloom-test", "", "Bloom过滤器验证：给定助记词，对比CPU与GPU bloom结果")
	enumTestMnem   = flag.String("enum-test", "", "枚举验证：给定完整助记词+模板，验证CPU/GPU索引转换是否正确（配合-template使用）")
	enumTemplate   = flag.String("template", "", "枚举验证用的模板（未知词用?，如 'word1 ? ? word4 ...'）")
	bip39DebugMnem = flag.String("bip39-debug", "", "BIP39 GPU调试：验证GPU checksum计算（需要-gpu）")
)

func main() {
	flag.Parse()

	// 独立模式：bloom 过滤器验证
	if *bloomTestMnem != "" {
		runBloomTest(*bloomTestMnem, *bloomFile)
		return
	}

	// 独立模式：BIP39 GPU checksum 调试
	if *bip39DebugMnem != "" {
		runBIP39Debug(*bip39DebugMnem)
		return
	}

	// 独立模式：枚举验证
	if *enumTestMnem != "" {
		runEnumTest(*enumTestMnem, *enumTemplate, *bloomFile)
		return
	}

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

	// 独立模式：全链路测速
	if *benchFullN > 0 {
		runBenchFull(*benchFullN, *useGPU, *workers)
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

	// 解析目标 GPU 设备列表
	deviceIDs := parseGPUDevices(*gpuDevices, *useGPU)

	log.Printf("========================================")
	log.Printf("  Boon Worker v2 (紧凑协议)")
	log.Printf("========================================")
	log.Printf("  ID:           %s", id)
	log.Printf("  调度服务器:    %s", *schedulerURL)
	log.Printf("  Bloom过滤:     %s", boolStr(bloomFilter != nil, "已加载", "未加载"))
	if len(deviceIDs) == 0 {
		log.Printf("  计算模式:      CPU (%d线程)", *workers)
	} else {
		log.Printf("  计算模式:      GPU x%d (设备 %v)", len(deviceIDs), deviceIDs)
	}
	log.Printf("========================================")

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

	switch len(deviceIDs) {
	case 0:
		// CPU 模式
		runCPUWorker(ctx, id, bloomFilter, *workers)

	case 1:
		// 单 GPU 模式（保留原有行为）
		runGPUWorkers(ctx, id, bloomFilter, deviceIDs)

	default:
		// 多 GPU 模式
		runGPUWorkers(ctx, id, bloomFilter, deviceIDs)
	}

	elapsed := time.Since(startTime)
	log.Printf("========================================")
	log.Printf("  Worker 已停止，运行时间: %v", elapsed)
	log.Printf("========================================")
}

// runCPUWorker 启动 CPU 模式 worker
func runCPUWorker(ctx context.Context, id string, bloomFilter *bloom.Filter, numWorkers int) {
	seedComp := compute.NewCPUComputer()
	client := worker.NewCompactClient(*schedulerURL)
	w := worker.NewCompactWorkerWithComputer(id, client, numWorkers, seedComp)
	w.SetBloomFilter(bloomFilter)
	w.Warmup()
	w.Start(ctx)
	<-ctx.Done()
	w.Stop()
}

// runGPUWorkers 启动一个或多个 GPU worker（每个设备一个独立 goroutine）
func runGPUWorkers(ctx context.Context, baseID string, bloomFilter *bloom.Filter, deviceIDs []int) {
	// 每个 GPU 分配的 CPU 枚举线程数
	cpuPerGPU := runtime.NumCPU() / len(deviceIDs)
	if cpuPerGPU < 1 {
		cpuPerGPU = 1
	}

	var wg sync.WaitGroup
	activeWorkers := make([]*worker.CompactWorker, 0, len(deviceIDs))
	var mu sync.Mutex

	for _, devID := range deviceIDs {
		gpu, err := compute.NewGPUComputer(devID)
		if err != nil {
			log.Printf("GPU %d 初始化失败，跳过: %v", devID, err)
			continue
		}

		wID := baseID
		if len(deviceIDs) > 1 {
			wID = fmt.Sprintf("%s-gpu%d", baseID, devID)
		}

		client := worker.NewCompactClient(*schedulerURL)
		w := worker.NewCompactWorkerWithComputer(wID, client, 1, gpu)
		w.SetBatchSize(65536)
		w.SetEnumWorkers(cpuPerGPU)
		w.SetBloomFilter(bloomFilter)

		mu.Lock()
		activeWorkers = append(activeWorkers, w)
		mu.Unlock()

		log.Printf("[GPU %d] Worker %s 初始化成功，CPU枚举线程: %d", devID, wID, cpuPerGPU)
	}

	if len(activeWorkers) == 0 {
		log.Printf("所有 GPU 初始化失败，退出")
		return
	}

	// 并行热身
	log.Printf("并行热身 %d 个 GPU worker...", len(activeWorkers))
	var warmupWg sync.WaitGroup
	for _, w := range activeWorkers {
		warmupWg.Add(1)
		go func(w *worker.CompactWorker) {
			defer warmupWg.Done()
			w.Warmup()
		}(w)
	}
	warmupWg.Wait()

	// 启动所有 worker
	for _, w := range activeWorkers {
		wg.Add(1)
		go func(w *worker.CompactWorker) {
			defer wg.Done()
			w.Start(ctx)
			<-ctx.Done()
			w.Stop()
		}(w)
	}

	wg.Wait()
}

// parseGPUDevices 解析 GPU 设备列表
// devStr: "" | "all" | "0,1,2"
// legacyGPU: -gpu 标志（等价于 "all"）
func parseGPUDevices(devStr string, legacyGPU bool) []int {
	if devStr == "" && !legacyGPU {
		return nil
	}

	total := compute.GPUDeviceCount()
	if total == 0 {
		log.Printf("未检测到 CUDA 设备（可能未使用 -tags cuda 编译）")
		return nil
	}

	if devStr == "" || devStr == "all" {
		ids := make([]int, total)
		for i := range ids {
			ids[i] = i
		}
		return ids
	}

	parts := strings.Split(devStr, ",")
	var ids []int
	for _, p := range parts {
		p = strings.TrimSpace(p)
		id, err := strconv.Atoi(p)
		if err != nil || id < 0 || id >= total {
			log.Printf("无效的GPU设备ID: %q（系统共 %d 个设备）", p, total)
			continue
		}
		ids = append(ids, id)
	}
	return ids
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
		g, err := compute.NewGPUComputer(0)
		if err != nil {
			log.Fatalf("GPU初始化失败: %v\n提示: 需要 CUDA 构建且有可用 GPU", err)
		}
		comp = g
		compName = "GPU(device 0)"
	} else {
		comp = compute.NewCPUComputer()
		compName = fmt.Sprintf("CPU(单线程)")
	}

	fmt.Printf("\n计算引擎: %s\n", compName)
	fmt.Printf("助记词总数: %d\n\n", total)

	batchSizes := []int{2048, 2048 << 1, 2048 << 2, 2048 << 3, 2048 << 4, 2048 << 5, 2048 << 6, 2048 << 7}

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
	gpuComp, err := compute.NewGPUComputer(0)
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

// ================================================================
// 全链路测速模式
// ================================================================

func runBenchFull(total int64, gpu bool, cpuWorkers int) {
	knownWords := []string{"abandon", "abandon", "abandon", "abandon", "abandon", "abandon", "abandon", "abandon", "", "", "", ""}

	var seedComp compute.SeedComputer
	compName := "CPU"
	if gpu {
		g, err := compute.NewGPUComputer(0)
		if err != nil {
			log.Fatalf("GPU初始化失败: %v\n提示: 需要 CUDA 构建且有可用 GPU", err)
		}
		seedComp = g
		compName = "GPU(device 0)"
	} else {
		seedComp = compute.NewCPUComputer()
	}

	fmt.Printf("全链路性能分析 (bench-full)\n")
	fmt.Printf("计算引擎: %s", compName)
	if !gpu {
		fmt.Printf(" (%d线程)", cpuWorkers)
	}
	fmt.Printf("\n模板: 8个已知词 + 4个未知词 (positions 8-11)\n")
	fmt.Printf("索引范围: [0, %d)\n\n", total)

	effectiveWorkers := cpuWorkers
	if gpu {
		effectiveWorkers = 1
	}
	batchSizes := []int64{500, 2048, 8192, 32768, 65536, 131072, 262144}
	if gpu {
		batchSizes = []int64{65536}
	}

	// 热身
	{
		warmupTask := &protocol.CompactTask{TaskID: 0, JobID: 1, StartIdx: 0, EndIdx: 2048}
		warmupEnum := worker.NewLocalEnumerator(&worker.TaskTemplate{
			JobID: 1, Words: append([]string(nil), knownWords...), UnknownPos: []int{8, 9, 10, 11},
		})
		cc := compute.NewCompactComputer(effectiveWorkers, seedComp)
		cc.SetBatchSize(2048)
		if gpu {
			cc.SetEnumWorkers(runtime.NumCPU())
		}
		cc.ComputeRange(warmupEnum, warmupTask, nil)
	}

	task := &protocol.CompactTask{TaskID: 1, JobID: 1, StartIdx: 0, EndIdx: total}

	fmt.Printf("  %-12s  %-10s  %-14s\n", "批次大小", "耗时", "速度(索引/s)")
	fmt.Printf("  %-12s  %-10s  %-14s\n", "------------", "----------", "--------------")

	for _, bs := range batchSizes {
		enum := worker.NewLocalEnumerator(&worker.TaskTemplate{
			JobID: 1, Words: append([]string(nil), knownWords...), UnknownPos: []int{8, 9, 10, 11},
		})
		cc := compute.NewCompactComputer(effectiveWorkers, seedComp)
		cc.SetBatchSize(bs)
		if gpu {
			cc.SetEnumWorkers(runtime.NumCPU())
		}

		start := time.Now()
		cc.ComputeRange(enum, task, nil)
		elapsed := time.Since(start)

		fmt.Printf("  %-12d  %-10s  %.0f\n", bs, fmtDuration(elapsed), float64(total)/elapsed.Seconds())
	}
	fmt.Println()
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

// ================================================================
// Bloom 过滤器验证模式
// ================================================================

type bloomUploader interface {
	UploadBloomFilter(f *bloom.Filter) error
}
type bloomTester interface {
	BloomTestAddr(addr []byte) bool
}

func runBloomTest(mnemonic, bloomFilePath string) {
	fmt.Printf("=== Bloom 过滤器验证 ===\n")
	fmt.Printf("助记词: %s\n", mnemonic)

	// 1. CPU 计算地址
	cpuComp := compute.NewCPUComputer()
	cpuAddrs := cpuComp.Compute([]string{mnemonic})
	if len(cpuAddrs) == 0 || len(cpuAddrs[0]) == 0 {
		log.Fatalf("CPU 计算地址失败")
	}
	cpuAddr := cpuAddrs[0]
	fmt.Printf("CPU 地址:  %s\n", hex.EncodeToString(cpuAddr))

	// 2. GPU 计算地址（使用设备 0）
	gpuCompConcrete, err := compute.NewGPUComputer(0)
	if err != nil {
		log.Fatalf("GPU 初始化失败: %v", err)
	}
	var gpuComp compute.SeedComputer = gpuCompConcrete
	gpuAddrs := gpuComp.Compute([]string{mnemonic})
	if len(gpuAddrs) == 0 || len(gpuAddrs[0]) == 0 {
		log.Fatalf("GPU 计算地址失败")
	}
	gpuAddr := gpuAddrs[0]
	fmt.Printf("GPU 地址:  %s\n", hex.EncodeToString(gpuAddr))

	if bytes.Equal(cpuAddr, gpuAddr) {
		fmt.Printf("地址对比:  ✓ CPU == GPU\n")
	} else {
		fmt.Printf("地址对比:  ✗ CPU != GPU  ← 地址计算 BUG\n")
	}

	// 3. 加载 bloom 过滤器
	bf, err := bloom.LoadFromFile(bloomFilePath)
	if err != nil {
		log.Fatalf("加载 bloom 文件失败: %v", err)
	}
	fmt.Printf("\nBloom 文件: %s\n", bloomFilePath)

	// 4. CPU bloom 检查
	cpuBloom := bf.Contains(cpuAddr)
	fmt.Printf("CPU bloom.Contains(cpu_addr):  %v\n", cpuBloom)

	// 5. 上传 bloom 到 GPU 并测试
	up, okUp := gpuComp.(bloomUploader)
	bt, okBt := gpuComp.(bloomTester)
	if !okUp || !okBt {
		log.Fatalf("当前构建不支持 GPU bloom 测试（需要 -tags cuda）")
	}
	if err := up.UploadBloomFilter(bf); err != nil {
		log.Fatalf("GPU bloom 上传失败: %v", err)
	}

	// 6. GPU bloom 测试 CPU 地址
	gpuBloomCPUAddr := bt.BloomTestAddr(cpuAddr)
	fmt.Printf("GPU bloom.test(cpu_addr):      %v\n", gpuBloomCPUAddr)

	// 7. GPU bloom 测试 GPU 地址（若地址不同则额外测试）
	if !bytes.Equal(cpuAddr, gpuAddr) {
		gpuBloomGPUAddr := bt.BloomTestAddr(gpuAddr)
		fmt.Printf("GPU bloom.test(gpu_addr):      %v\n", gpuBloomGPUAddr)
	}

	fmt.Printf("\n=== 诊断 ===\n")
	switch {
	case !cpuBloom:
		fmt.Printf("❌ 地址不在 bloom 过滤器中（CPU），请确认助记词/bloom文件是否匹配\n")
	case cpuBloom && !gpuBloomCPUAddr:
		fmt.Printf("❌ GPU bloom hash 与 Go 实现不一致 → GPU bloom 过滤器 BUG\n")
	case cpuBloom && gpuBloomCPUAddr:
		fmt.Printf("✓ CPU 和 GPU bloom 均通过，bloom 过滤器实现正确\n")
		fmt.Printf("  若枚举时仍漏报，问题可能在枚举索引→助记词的转换逻辑\n")
	}
}

// ================================================================
// BIP39 GPU checksum 调试
// ================================================================

type bip39Debugger interface {
	BIP39Debug(wordIndices [12]int16) (storedCS, sha0 byte, err error)
}

func runBIP39Debug(mnemonic string) {
	fmt.Printf("=== GPU BIP39 Checksum 调试 ===\n")
	words := strings.Fields(mnemonic)
	if len(words) != 12 {
		log.Fatalf("助记词必须是12个词，当前: %d", len(words))
	}
	wordToIdx := make(map[string]int)
	for i, w := range mn.WordList {
		wordToIdx[w] = i
	}
	var wi [12]int16
	for i, w := range words {
		idx, ok := wordToIdx[w]
		if !ok {
			log.Fatalf("词 '%s' 不在BIP39词表中", w)
		}
		wi[i] = int16(idx)
		fmt.Printf("  wi[%d] = %d (%s)\n", i, idx, w)
	}

	gpuRaw, err := compute.NewGPUComputer(0)
	if err != nil {
		log.Fatalf("GPU 初始化失败: %v", err)
	}
	defer gpuRaw.Close()

	dbg, ok := any(gpuRaw).(bip39Debugger)
	if !ok {
		log.Fatalf("当前构建不支持 GPU BIP39 调试")
	}
	storedCS, sha0, err := dbg.BIP39Debug(wi)
	if err != nil {
		log.Fatalf("BIP39Debug 失败: %v", err)
	}
	fmt.Printf("\nGPU stored_cs (bits & 0xF): 0x%X\n", storedCS)
	fmt.Printf("GPU sha256[0]:              0x%02X\n", sha0)
	fmt.Printf("GPU sha256[0] >> 4:         0x%X\n", sha0>>4)
	if storedCS == sha0>>4 {
		fmt.Printf("✓ GPU checksum 通过\n")
	} else {
		fmt.Printf("✗ GPU checksum 不匹配 → BIP39 filter 会拒绝该助记词\n")
	}
}

// ================================================================
// 枚举验证模式
// ================================================================

type gpuEnumerator interface {
	EnumerateCompute(startIdx, endIdx int64, knownWordIndices []int16, unknownPositions []int8) ([]int64, [][]byte, error)
}

func runEnumTest(targetMnemonic, template, bloomFilePath string) {
	fmt.Printf("=== 枚举验证 ===\n")
	fmt.Printf("目标助记词: %s\n", targetMnemonic)

	targetWords := strings.Fields(targetMnemonic)
	if len(targetWords) != 12 {
		log.Fatalf("助记词必须是12个词，当前: %d", len(targetWords))
	}

	// 构造模板词数组（? 或空位 = 未知）
	var tmplWords []string
	if template == "" {
		// 默认：最后4个词未知（避免 int64 溢出）
		tmplWords = make([]string, 12)
		copy(tmplWords, targetWords[:8])
		fmt.Printf("模板: 未指定，默认前8个词已知，后4个词未知\n")
	} else {
		tmplWords = strings.Fields(template)
		if len(tmplWords) != 12 {
			log.Fatalf("模板必须是12个词，当前: %d", len(tmplWords))
		}
		for i, w := range tmplWords {
			if w == "?" {
				tmplWords[i] = ""
			}
		}
		fmt.Printf("模板: %s\n", template)
	}

	// 确认已知词匹配，收集未知位置
	unknownPos := make([]int, 0)
	for i, w := range tmplWords {
		if w == "" {
			unknownPos = append(unknownPos, i)
		} else if w != targetWords[i] {
			log.Fatalf("模板第%d位词'%s'与目标词'%s'不匹配", i, w, targetWords[i])
		}
	}
	fmt.Printf("未知位置: %v (%d个)\n", unknownPos, len(unknownPos))

	// 检测溢出：未知词数超过6个时 2048^n > int64_max
	if len(unknownPos) > 6 {
		log.Fatalf("未知词数 %d 超过6个，枚举索引会溢出 int64。\n请用 -template 指定更多已知词，例如：\n  -template \"%s ? ? ? ?\"",
			len(unknownPos), strings.Join(targetWords[:8], " "))
	}

	// 计算目标助记词的枚举索引
	wordToIdx := make(map[string]int)
	for i, w := range mn.WordList {
		wordToIdx[w] = i
	}
	var targetIdx int64
	var multiplier int64 = 1
	for _, pos := range unknownPos {
		wi, ok := wordToIdx[targetWords[pos]]
		if !ok {
			log.Fatalf("词 '%s' 不在BIP39词表中", targetWords[pos])
		}
		targetIdx += int64(wi) * multiplier
		multiplier *= 2048
	}
	fmt.Printf("目标枚举索引: %d\n\n", targetIdx)

	// CPU EnumerateAt 验证
	tmpl := &worker.TaskTemplate{
		JobID:      0,
		Words:      tmplWords,
		UnknownPos: unknownPos,
	}
	enum := worker.NewLocalEnumerator(tmpl)
	validator := mn.NewValidator()
	cpuWords, valid := enum.EnumerateAt(targetIdx, validator)
	fmt.Printf("CPU EnumerateAt(%d):\n", targetIdx)
	if !valid {
		fmt.Printf("  ✗ BIP39 校验失败（checksum 无效）\n")
		cpuWords = targetWords
	} else if strings.Join(cpuWords, " ") == targetMnemonic {
		fmt.Printf("  ✓ 正确还原: %s\n", strings.Join(cpuWords, " "))
	} else {
		fmt.Printf("  ✗ 还原错误:\n    得到: %s\n    期望: %s\n", strings.Join(cpuWords, " "), targetMnemonic)
	}

	cpuComp := compute.NewCPUComputer()
	cpuAddr := cpuComp.Compute([]string{strings.Join(cpuWords, " ")})[0]
	fmt.Printf("  CPU 地址: %s\n\n", hex.EncodeToString(cpuAddr))

	// GPU 枚举（使用设备 0，通过接口断言访问 EnumerateCompute）
	gpuRaw, err := compute.NewGPUComputer(0)
	if err != nil {
		log.Fatalf("GPU 初始化失败: %v", err)
	}
	var gpuSeed compute.SeedComputer = gpuRaw
	ge, okGE := gpuSeed.(gpuEnumerator)
	if !okGE {
		log.Fatalf("当前构建不支持 GPU 枚举（需要 -tags cuda）")
	}

	knownWordIndices, unknownPositions := enum.TemplateIndices()

	fmt.Printf("GPU EnumerateCompute([%d, %d)) 无bloom:\n", targetIdx, targetIdx+1)
	idxs, addrs, err := ge.EnumerateCompute(targetIdx, targetIdx+1, knownWordIndices, unknownPositions)
	if err != nil {
		fmt.Printf("  ✗ GPU 枚举出错: %v\n", err)
	} else if len(idxs) == 0 {
		fmt.Printf("  ✗ 未返回结果 → GPU BIP39 checksum 将该助记词过滤掉了\n")
	} else {
		gpuAddr := addrs[0]
		fmt.Printf("  GPU 地址: %s  (idx=%d)\n", hex.EncodeToString(gpuAddr), idxs[0])
		if bytes.Equal(cpuAddr, gpuAddr) {
			fmt.Printf("  ✓ CPU == GPU\n")
		} else {
			fmt.Printf("  ✗ CPU != GPU ← GPU 推导 BUG\n")
		}
	}

	if bloomFilePath == "" {
		return
	}
	bf, err := bloom.LoadFromFile(bloomFilePath)
	if err != nil {
		fmt.Printf("\n加载 bloom 失败: %v\n", err)
		return
	}
	up, okUp := gpuSeed.(bloomUploader)
	if !okUp {
		return
	}
	if err := up.UploadBloomFilter(bf); err != nil {
		fmt.Printf("\nGPU bloom 上传失败: %v\n", err)
		return
	}

	fmt.Printf("\nGPU EnumerateCompute([%d, %d)) + bloom:\n", targetIdx, targetIdx+1)
	idxsB, addrsB, err := ge.EnumerateCompute(targetIdx, targetIdx+1, knownWordIndices, unknownPositions)
	if err != nil {
		fmt.Printf("  ✗ GPU 枚举出错: %v\n", err)
	} else if len(idxsB) == 0 {
		cpuBloom := bf.Contains(cpuAddr)
		fmt.Printf("  ✗ GPU bloom 过滤掉了该地址\n")
		fmt.Printf("  CPU bloom.Contains: %v\n", cpuBloom)
		if cpuBloom {
			fmt.Printf("  → GPU bloom hash 实现 BUG\n")
		} else {
			fmt.Printf("  → 地址不在 bloom 中（bloom 文件不包含该地址）\n")
		}
	} else {
		fmt.Printf("  ✓ bloom 通过: %s\n", hex.EncodeToString(addrsB[0]))
	}
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
