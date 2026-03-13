package worker

import (
	"context"
	"log"
	"sync"
	"sync/atomic"
	"time"

	"boon/internal/bloom"
	"boon/internal/compute"
	"boon/internal/protocol"
)

// pendingTask 带模板的待处理任务（每次从服务器获取，不在本地存储）
type pendingTask struct {
	task     *protocol.CompactTask
	template *TaskTemplate
}

// CompactWorker 紧凑协议Worker
type CompactWorker struct {
	id          string
	client      *CompactClient
	computer    *compute.CompactComputer
	bloomFilter *bloom.Filter // 本地Bloom过滤器（可选）

	pollInterval time.Duration
	workers      int

	taskQueue chan *pendingTask
	stopCh    chan struct{}
	wg        sync.WaitGroup

	running int32

	// 统计
	stats struct {
		tasksFetched   int64
		tasksComputed  int64
		matchesFound   int64
		indicesScanned int64
		currentSpeed   int64 // 当前枚举速度（indices/s），上报给调度器用于动态分配
	}
}

// NewCompactWorker 创建紧凑Worker（使用CPU计算引擎）
func NewCompactWorker(id string, client *CompactClient, workers int) *CompactWorker {
	return NewCompactWorkerWithComputer(id, client, workers, compute.NewCPUComputer())
}

// NewCompactWorkerWithComputer 创建紧凑Worker（使用指定计算引擎）
func NewCompactWorkerWithComputer(id string, client *CompactClient, workers int, seedComp compute.SeedComputer) *CompactWorker {
	return &CompactWorker{
		id:           id,
		client:       client,
		computer:     compute.NewCompactComputer(workers, seedComp),
		pollInterval: 50 * time.Millisecond,
		workers:      workers,
		taskQueue:    make(chan *pendingTask, 2),
		stopCh:       make(chan struct{}),
	}
}

// SetBloomFilter 设置Bloom过滤器
func (w *CompactWorker) SetBloomFilter(filter *bloom.Filter) {
	w.bloomFilter = filter
}

// Start 启动Worker
func (w *CompactWorker) Start(ctx context.Context) {
	if !atomic.CompareAndSwapInt32(&w.running, 0, 1) {
		return
	}

	// 启动任务拉取器
	w.wg.Add(1)
	go w.runFetcher(ctx)

	// 启动计算worker（1个即可，CompactComputer内部已经并行处理）
	w.wg.Add(1)
	go w.runComputer(ctx)

	// 启动统计
	w.wg.Add(1)
	go w.printStats(ctx)
}

// Stop 停止Worker
func (w *CompactWorker) Stop() {
	if !atomic.CompareAndSwapInt32(&w.running, 1, 0) {
		return
	}
	close(w.stopCh)
	w.wg.Wait()
}

// runFetcher 任务拉取器
func (w *CompactWorker) runFetcher(ctx context.Context) {
	defer w.wg.Done()

	ticker := time.NewTicker(w.pollInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-w.stopCh:
			return
		case <-ticker.C:
			queueLen := len(w.taskQueue)
			prefetch := cap(w.taskQueue)
			if queueLen < prefetch {
				w.fetchTasks(ctx, prefetch-queueLen)
			}
		}
	}
}

// fetchTasks 拉取任务，携带当前速度，每次都从服务器获取模板，不做本地缓存
func (w *CompactWorker) fetchTasks(ctx context.Context, count int) {
	for i := 0; i < count; i++ {
		select {
		case <-ctx.Done():
			return
		default:
		}

		// 携带当前枚举速度，让调度器动态决定分配多大的枚举空间
		speed := atomic.LoadInt64(&w.stats.currentSpeed)
		task, err := w.client.FetchTask(w.id, speed)
		if err != nil {
			log.Printf("[Worker %s] 拉取任务失败: %v", w.id, err)
			time.Sleep(100 * time.Millisecond)
			continue
		}
		if task == nil {
			break // 当前没有可用任务
		}

		// 每次都从服务器拉取模板，不在本地存储
		tmpl, err := w.client.FetchTemplate(task.JobID)
		if err != nil {
			log.Printf("[Worker %s] 获取模板失败: %v", w.id, err)
			continue
		}
		if tmpl == nil {
			log.Printf("[Worker %s] 模板暂不可用，稍后重试", w.id)
			continue
		}

		select {
		case w.taskQueue <- &pendingTask{task: task, template: tmpl}:
			atomic.AddInt64(&w.stats.tasksFetched, 1)
		case <-ctx.Done():
			return
		}
	}
}

// runComputer 计算worker
func (w *CompactWorker) runComputer(ctx context.Context) {
	defer w.wg.Done()

	for {
		select {
		case <-ctx.Done():
			return
		case <-w.stopCh:
			return
		case pt, ok := <-w.taskQueue:
			if !ok {
				return
			}
			w.processTask(ctx, pt)
		}
	}
}

// processTask 处理任务，枚举器从模板即时创建，用完即丢，完成后更新速度
func (w *CompactWorker) processTask(ctx context.Context, pt *pendingTask) {
	start := time.Now()
	indices := pt.task.EndIdx - pt.task.StartIdx

	// 从本次任务的模板创建枚举器（不存储，用完即丢）
	enum := NewLocalEnumerator(pt.template)

	// 创建Bloom过滤函数
	var bloomFunc func([]byte) bool
	if w.bloomFilter != nil {
		bloomFunc = func(addr []byte) bool {
			return w.bloomFilter.Contains(addr)
		}
	}

	// 计算范围内的匹配
	result := w.computer.ComputeRange(enum, pt.task, bloomFunc)

	elapsed := time.Since(start)

	atomic.AddInt64(&w.stats.tasksComputed, 1)
	atomic.AddInt64(&w.stats.indicesScanned, indices)

	// 更新枚举速度（EWMA，平滑抖动），供下次握手上报给调度器
	if elapsed > 0 {
		newSpeed := int64(float64(indices) / elapsed.Seconds())
		oldSpeed := atomic.LoadInt64(&w.stats.currentSpeed)
		var smoothed int64
		if oldSpeed == 0 {
			smoothed = newSpeed
		} else {
			smoothed = int64(float64(oldSpeed)*0.8 + float64(newSpeed)*0.2)
		}
		atomic.StoreInt64(&w.stats.currentSpeed, smoothed)
	}

	// 提交结果
	for retry := 0; retry < 3; retry++ {
		err := w.client.SubmitResult(w.id, result)
		if err == nil {
			atomic.AddInt64(&w.stats.matchesFound, int64(len(result.Matches)))
			rate := float64(indices) / elapsed.Seconds()
			if len(result.Matches) > 0 {
				log.Printf("[Worker %s] 匹配! task=%d matches=%d 速率=%.0f/s",
					w.id, pt.task.TaskID, len(result.Matches), rate)
			}
			return
		}
		log.Printf("[Worker %s] 提交失败: %v", w.id, err)
		time.Sleep(time.Second)
	}
}

// printStats 打印统计
func (w *CompactWorker) printStats(ctx context.Context) {
	defer w.wg.Done()

	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-w.stopCh:
			return
		case <-ticker.C:
			computed := atomic.LoadInt64(&w.stats.tasksComputed)
			scanned := atomic.LoadInt64(&w.stats.indicesScanned)
			matches := atomic.LoadInt64(&w.stats.matchesFound)
			speed := atomic.LoadInt64(&w.stats.currentSpeed)
			queue := len(w.taskQueue)

			log.Printf("[Worker %s] 任务=%d 扫描=%d 匹配=%d 速度=%d/s 队列=%d",
				w.id, computed, scanned, matches, speed, queue)
		}
	}
}
