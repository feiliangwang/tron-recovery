package worker

import (
	"boon/internal/compute"
	"boon/internal/protocol"
	"context"
	"log"
	"sync"
	"sync/atomic"
	"time"
)

// SchedulerClient 调度器客户端接口
type SchedulerClient interface {
	FetchTask(workerID string) (*protocol.Task, error)
	SubmitResult(workerID string, result *protocol.Result) error
}

// Worker 计算节点
type Worker struct {
	id       string
	computer compute.SeedComputer
	client   SchedulerClient

	pollInterval   time.Duration
	computeWorkers int
	prefetchSize   int // 预取任务数量

	taskQueue chan *protocol.Task
	stopCh    chan struct{}
	wg        sync.WaitGroup

	running int32

	// 统计
	stats struct {
		tasksFetched   int64
		tasksComputed  int64
		tasksSubmitted int64
	}
}

// NewWorker 创建计算节点
func NewWorker(id string, client SchedulerClient, computer compute.SeedComputer, workers int) *Worker {
	return &Worker{
		id:             id,
		computer:       computer,
		client:         client,
		pollInterval:   50 * time.Millisecond,
		computeWorkers: workers,
		prefetchSize:   workers * 2, // 预取任务数为worker的2倍
		taskQueue:      make(chan *protocol.Task, workers*3),
		stopCh:         make(chan struct{}),
	}
}

// SetPollInterval 设置轮询间隔
func (w *Worker) SetPollInterval(d time.Duration) {
	w.pollInterval = d
}

// SetPrefetchSize 设置预取任务数
func (w *Worker) SetPrefetchSize(size int) {
	w.prefetchSize = size
}

// Start 启动worker
func (w *Worker) Start(ctx context.Context) {
	if !atomic.CompareAndSwapInt32(&w.running, 0, 1) {
		return // 已经在运行
	}

	// 启动任务拉取器
	w.wg.Add(1)
	go w.runFetcher(ctx)

	// 启动计算worker
	for i := 0; i < w.computeWorkers; i++ {
		w.wg.Add(1)
		go w.runComputer(ctx, i)
	}

	// 启动统计打印
	w.wg.Add(1)
	go w.printStats(ctx)
}

// Stop 停止worker
func (w *Worker) Stop() {
	if !atomic.CompareAndSwapInt32(&w.running, 1, 0) {
		return
	}
	close(w.stopCh)
	w.wg.Wait()
}

// runFetcher 任务拉取器 - 持续获取任务保持队列满
func (w *Worker) runFetcher(ctx context.Context) {
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
			// 检查队列是否需要补充
			queueLen := len(w.taskQueue)
			if queueLen < w.prefetchSize {
				// 需要补充任务
				w.fetchTasks(ctx, w.prefetchSize-queueLen)
			}
		}
	}
}

// fetchTasks 拉取任务
func (w *Worker) fetchTasks(ctx context.Context, count int) {
	fetched := 0
	for i := 0; i < count; i++ {
		select {
		case <-ctx.Done():
			return
		default:
		}

		task, err := w.client.FetchTask(w.id)
		if err != nil {
			// 网络错误，短暂等待后重试
			time.Sleep(100 * time.Millisecond)
			continue
		}

		if task == nil {
			// 没有更多任务
			break
		}

		select {
		case w.taskQueue <- task:
			fetched++
			atomic.AddInt64(&w.stats.tasksFetched, 1)
		case <-ctx.Done():
			return
		}
	}
}

// runComputer 计算worker
func (w *Worker) runComputer(ctx context.Context, workerID int) {
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

// processTask 处理单个任务
func (w *Worker) processTask(ctx context.Context, task *protocol.Task) {
	start := time.Now()

	// 计算地址
	addresses := w.computer.Compute(task.Mnemonics)
	atomic.AddInt64(&w.stats.tasksComputed, 1)

	// 构建结果
	result := &protocol.Result{
		TaskID:    task.ID,
		Addresses: addresses,
		Mnemonics: task.Mnemonics,
	}

	// 提交结果（带重试）
	for retry := 0; retry < 3; retry++ {
		err := w.client.SubmitResult(w.id, result)
		if err == nil {
			atomic.AddInt64(&w.stats.tasksSubmitted, 1)
			elapsed := time.Since(start)
			if elapsed > time.Second {
				log.Printf("[Worker %s] 任务完成 [task=%d] 耗时=%v 队列=%d",
					w.id, task.ID, elapsed, len(w.taskQueue))
			}
			return
		}

		log.Printf("[Worker %s] 提交失败 [task=%d] 重试=%d: %v", w.id, task.ID, retry+1, err)
		time.Sleep(time.Second)
	}

	log.Printf("[Worker %s] 提交失败 [task=%d] 放弃", w.id, task.ID)
}

// printStats 打印统计
func (w *Worker) printStats(ctx context.Context) {
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
			fetched := atomic.LoadInt64(&w.stats.tasksFetched)
			computed := atomic.LoadInt64(&w.stats.tasksComputed)
			submitted := atomic.LoadInt64(&w.stats.tasksSubmitted)
			queueLen := len(w.taskQueue)

			log.Printf("[Worker %s] 统计: 拉取=%d 计算=%d 提交=%d 队列=%d",
				w.id, fetched, computed, submitted, queueLen)
		}
	}
}

// GetStats 获取统计信息
func (w *Worker) GetStats() map[string]int64 {
	return map[string]int64{
		"tasks_fetched":   atomic.LoadInt64(&w.stats.tasksFetched),
		"tasks_computed":  atomic.LoadInt64(&w.stats.tasksComputed),
		"tasks_submitted": atomic.LoadInt64(&w.stats.tasksSubmitted),
		"queue_length":    int64(len(w.taskQueue)),
	}
}
