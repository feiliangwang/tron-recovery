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

// CompactWorker 紧凑协议Worker
type CompactWorker struct {
	id          string
	client      *CompactClient
	computer    *compute.CompactComputer
	bloomFilter *bloom.Filter // 本地Bloom过滤器

	pollInterval time.Duration
	workers      int

	template   *TaskTemplate
	enumerator *LocalEnumerator

	taskQueue chan *protocol.CompactTask
	stopCh    chan struct{}
	wg        sync.WaitGroup

	running int32

	// 统计
	stats struct {
		tasksFetched   int64
		tasksComputed  int64
		matchesFound   int64
		indicesScanned int64
	}
}

// NewCompactWorker 创建紧凑Worker
func NewCompactWorker(id string, client *CompactClient, workers int) *CompactWorker {
	return &CompactWorker{
		id:           id,
		client:       client,
		computer:     compute.NewCompactComputer(workers),
		pollInterval: 50 * time.Millisecond,
		workers:      workers,
		taskQueue:    make(chan *protocol.CompactTask, workers*2),
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

	// 启动计算worker
	for i := 0; i < w.workers; i++ {
		w.wg.Add(1)
		go w.runComputer(ctx)
	}

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

// SetTemplate 设置任务模板
func (w *CompactWorker) SetTemplate(template *TaskTemplate) {
	w.template = template
	w.enumerator = NewLocalEnumerator(template)
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

// fetchTasks 拉取任务
func (w *CompactWorker) fetchTasks(ctx context.Context, count int) {
	for i := 0; i < count; i++ {
		select {
		case <-ctx.Done():
			return
		default:
		}

		task, err := w.client.FetchTask(w.id)
		if err != nil {
			time.Sleep(100 * time.Millisecond)
			continue
		}

		if task == nil {
			break
		}

		// 获取模板（如果需要）
		if w.enumerator == nil {
			tmpl, err := w.client.FetchTemplate(task.JobID)
			if err != nil {
				log.Printf("[Worker %s] 获取模板失败: %v", w.id, err)
				continue
			}
			w.SetTemplate(tmpl)
			log.Printf("[Worker %s] 模板加载完成, 未知位置: %v", w.id, tmpl.UnknownPos)
		}

		select {
		case w.taskQueue <- task:
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
		case task, ok := <-w.taskQueue:
			if !ok {
				return
			}
			w.processTask(ctx, task)
		}
	}
}

// processTask 处理任务
func (w *CompactWorker) processTask(ctx context.Context, task *protocol.CompactTask) {
	if w.enumerator == nil {
		return
	}

	start := time.Now()

	// 创建Bloom过滤函数
	var bloomFunc func([]byte) bool
	if w.bloomFilter != nil {
		bloomFunc = func(addr []byte) bool {
			return w.bloomFilter.Contains(addr)
		}
	}

	// 计算范围内的匹配
	result := w.computer.ComputeRange(w.enumerator, task, bloomFunc)

	atomic.AddInt64(&w.stats.tasksComputed, 1)
	atomic.AddInt64(&w.stats.indicesScanned, task.EndIdx-task.StartIdx)

	// 提交结果
	for retry := 0; retry < 3; retry++ {
		err := w.client.SubmitResult(w.id, result)
		if err == nil {
			atomic.AddInt64(&w.stats.matchesFound, int64(len(result.Matches)))
			elapsed := time.Since(start)
			rate := float64(task.EndIdx-task.StartIdx) / elapsed.Seconds()
			if len(result.Matches) > 0 {
				log.Printf("[Worker %s] 匹配! task=%d matches=%d 速率=%.0f/s",
					w.id, task.TaskID, len(result.Matches), rate)
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
			queue := len(w.taskQueue)

			log.Printf("[Worker %s] 任务=%d 扫描=%d 匹配=%d 队列=%d",
				w.id, computed, scanned, matches, queue)
		}
	}
}
