package scheduler

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/syndtr/goleveldb/leveldb"
	"github.com/syndtr/goleveldb/leveldb/util"
)

// Job 任务作业
type Job struct {
	ID          string     `json:"id"`
	Name        string     `json:"name"`
	Mnemonic    string     `json:"mnemonic"`
	BatchSize   int        `json:"batch_size"`
	Status      string     `json:"status"` // pending, running, paused, completed
	Total       int64      `json:"total"`
	Completed   int64      `json:"completed"`
	Matches     int64      `json:"matches"`
	CreatedAt   time.Time  `json:"created_at"`
	StartedAt   *time.Time `json:"started_at"`
	CompletedAt *time.Time `json:"completed_at"`
}

// PendingTask 飞行中任务记录
type PendingTask struct {
	JobID     string `json:"job_id"`
	StartIdx  int64  `json:"start_idx"`
	EndIdx    int64  `json:"end_idx"`
	TaskID    int64  `json:"task_id"`
	FetchedAt int64  `json:"fetched_at"` // Unix timestamp
}

// TaskManager 任务管理器
type TaskManager struct {
	db     *leveldb.DB
	jobs   map[string]*Job
	mu     sync.RWMutex
	autoID int
}

// Key prefixes
var (
	keyPrefixJob     = []byte("job:")
	keyPrefixPending = []byte("pending:") // pending:{taskID}
	keyMetaAutoID    = []byte("meta:autoID")
)

// NewTaskManager 创建任务管理器
func NewTaskManager(dataDir string) (*TaskManager, error) {
	dbPath := filepath.Join(dataDir, "scheduler.db")
	if err := os.MkdirAll(dataDir, 0755); err != nil {
		return nil, err
	}

	db, err := leveldb.OpenFile(dbPath, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to open leveldb: %w", err)
	}

	tm := &TaskManager{
		db:   db,
		jobs: make(map[string]*Job),
	}

	// 恢复 autoID
	if data, err := db.Get(keyMetaAutoID, nil); err == nil {
		tm.autoID = int(binary.BigEndian.Uint64(data))
	}

	// 加载所有 jobs
	tm.loadJobs()

	return tm, nil
}

// loadJobs 加载所有任务
func (tm *TaskManager) loadJobs() {
	iter := tm.db.NewIterator(util.BytesPrefix(keyPrefixJob), nil)
	defer iter.Release()

	for iter.Next() {
		var job Job
		if err := json.Unmarshal(iter.Value(), &job); err != nil {
			continue
		}
		tm.jobs[job.ID] = &job

		// 恢复 autoID
		var num int
		if _, err := fmt.Sscanf(job.ID, "job-%d", &num); err == nil && num > tm.autoID {
			tm.autoID = num
		}
	}
}

// Close 关闭数据库
func (tm *TaskManager) Close() error {
	return tm.db.Close()
}

// CreateJob 创建任务
func (tm *TaskManager) CreateJob(name, mnemonic string, batchSize int) *Job {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	tm.autoID++
	id := fmt.Sprintf("job-%d", tm.autoID)

	job := &Job{
		ID:        id,
		Name:      name,
		Mnemonic:  mnemonic,
		BatchSize: batchSize,
		Status:    "pending",
		CreatedAt: time.Now(),
	}

	tm.jobs[id] = job
	tm.saveJob(job)
	tm.saveAutoID()

	return job
}

// GetJob 获取任务
func (tm *TaskManager) GetJob(id string) *Job {
	tm.mu.RLock()
	defer tm.mu.RUnlock()
	return tm.jobs[id]
}

// ListJobs 列出所有任务
func (tm *TaskManager) ListJobs() []*Job {
	tm.mu.RLock()
	defer tm.mu.RUnlock()

	jobs := make([]*Job, 0, len(tm.jobs))
	for _, job := range tm.jobs {
		jobs = append(jobs, job)
	}
	return jobs
}

// UpdateJob 更新任务
func (tm *TaskManager) UpdateJob(job *Job) {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	if _, exists := tm.jobs[job.ID]; exists {
		tm.jobs[job.ID] = job
		tm.saveJob(job)
	}
}

// StartJob 启动任务
func (tm *TaskManager) StartJob(id string) bool {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	job, exists := tm.jobs[id]
	if !exists || job.Status == "completed" {
		return false
	}

	now := time.Now()
	job.Status = "running"
	job.StartedAt = &now
	tm.saveJob(job)
	return true
}

// PauseJob 暂停任务
func (tm *TaskManager) PauseJob(id string) bool {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	job, exists := tm.jobs[id]
	if !exists || job.Status != "running" {
		return false
	}

	job.Status = "paused"
	tm.saveJob(job)
	return true
}

// ResumeJob 恢复任务
func (tm *TaskManager) ResumeJob(id string) bool {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	job, exists := tm.jobs[id]
	if !exists || job.Status != "paused" {
		return false
	}

	job.Status = "running"
	tm.saveJob(job)
	return true
}

// DeleteJob 删除任务
func (tm *TaskManager) DeleteJob(id string) bool {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	if _, exists := tm.jobs[id]; !exists {
		return false
	}

	delete(tm.jobs, id)

	// 删除 job 数据
	if err := tm.db.Delete(tm.jobKey(id), nil); err != nil {
		log.Printf("[TaskManager] 删除 job 失败: %v", err)
	}
	// 删除该 job 的所有 pending tasks
	tm.deletePendingTasksByJob(id)

	return true
}

// CompleteJob 完成任务
func (tm *TaskManager) CompleteJob(id string) {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	if job, exists := tm.jobs[id]; exists {
		now := time.Now()
		job.Status = "completed"
		job.CompletedAt = &now
		tm.saveJob(job)
		// 删除该 job 的所有 pending tasks
		tm.deletePendingTasksByJob(id)
	}
}

// SetTotal 设置总数
func (tm *TaskManager) SetTotal(id string, total int64) {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	if job, exists := tm.jobs[id]; exists {
		job.Total = total
		tm.saveJob(job)
	}
}

// SetCompleted 设置已完成数（由 worker 提交结果时更新）
func (tm *TaskManager) SetCompleted(id string, completed int64) {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	if job, exists := tm.jobs[id]; exists {
		job.Completed = completed
		tm.saveJob(job)
	}
}

// IncrementMatches 增加匹配计数
func (tm *TaskManager) IncrementMatches(id string, count int) {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	if job, exists := tm.jobs[id]; exists {
		job.Matches += int64(count)
		tm.saveJob(job)
	}
}

// ============================================================
// Pending Tasks 管理（飞行中任务持久化）
// ============================================================

// AddPendingTask 添加飞行中任务（分发时调用）
func (tm *TaskManager) AddPendingTask(pt PendingTask) {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	data, err := json.Marshal(pt)
	if err != nil {
		log.Printf("[TaskManager] 序列化 pending task 失败: %v", err)
		return
	}

	key := tm.pendingKey(pt.TaskID)
	if err := tm.db.Put(key, data, nil); err != nil {
		log.Printf("[TaskManager] 保存 pending task 失败: %v", err)
	}
}

// CompletePendingTask 完成飞行中任务（提交时调用）
func (tm *TaskManager) CompletePendingTask(taskID int64) {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	key := tm.pendingKey(taskID)
	if err := tm.db.Delete(key, nil); err != nil {
		log.Printf("[TaskManager] 删除 pending task 失败: %v", err)
	}
}

// LoadAllPendingTasks 加载所有飞行中任务（重启恢复时调用）
func (tm *TaskManager) LoadAllPendingTasks() []PendingTask {
	tm.mu.RLock()
	defer tm.mu.RUnlock()

	return tm.loadAllPendingTasksLocked()
}

// loadAllPendingTasksLocked 内部方法，不加锁
func (tm *TaskManager) loadAllPendingTasksLocked() []PendingTask {
	var tasks []PendingTask
	iter := tm.db.NewIterator(util.BytesPrefix(keyPrefixPending), nil)
	defer iter.Release()

	for iter.Next() {
		var pt PendingTask
		if err := json.Unmarshal(iter.Value(), &pt); err != nil {
			continue
		}
		tasks = append(tasks, pt)
	}
	return tasks
}

// GetPendingTasksByJob 获取指定 job 的飞行中任务
func (tm *TaskManager) GetPendingTasksByJob(jobID string) []PendingTask {
	tm.mu.RLock()
	defer tm.mu.RUnlock()

	all := tm.loadAllPendingTasksLocked()
	var result []PendingTask
	for _, pt := range all {
		if pt.JobID == jobID {
			result = append(result, pt)
		}
	}
	return result
}

// deletePendingTasksByJob 删除指定 job 的所有 pending tasks（调用前须持有锁）
func (tm *TaskManager) deletePendingTasksByJob(jobID string) {
	iter := tm.db.NewIterator(util.BytesPrefix(keyPrefixPending), nil)
	var keysToDelete [][]byte
	for iter.Next() {
		var pt PendingTask
		if err := json.Unmarshal(iter.Value(), &pt); err != nil {
			continue
		}
		if pt.JobID == jobID {
			keysToDelete = append(keysToDelete, append([]byte{}, iter.Key()...))
		}
	}
	iter.Release()

	for _, key := range keysToDelete {
		tm.db.Delete(key, nil)
	}
}

// ============================================================
// 内部方法
// ============================================================

func (tm *TaskManager) jobKey(id string) []byte {
	return append([]byte("job:"), []byte(id)...)
}

func (tm *TaskManager) pendingKey(taskID int64) []byte {
	key := make([]byte, 8+8)
	copy(key[0:8], "pending:")
	binary.BigEndian.PutUint64(key[8:16], uint64(taskID))
	return key
}

func (tm *TaskManager) saveJob(job *Job) {
	data, err := json.Marshal(job)
	if err != nil {
		log.Printf("[TaskManager] 序列化 job 失败: %v", err)
		return
	}
	if err := tm.db.Put(tm.jobKey(job.ID), data, nil); err != nil {
		log.Printf("[TaskManager] 保存 job 失败: %v", err)
	}
}

func (tm *TaskManager) saveAutoID() {
	data := make([]byte, 8)
	binary.BigEndian.PutUint64(data, uint64(tm.autoID))
	if err := tm.db.Put(keyMetaAutoID, data, nil); err != nil {
		log.Printf("[TaskManager] 保存 autoID 失败: %v", err)
	}
}

// ============================================================
// Match 记录管理（存储在 LevelDB，key=match:{address}）
// ============================================================

// MatchRecord 匹配记录
type MatchRecord struct {
	JobID             string    `json:"job_id"`
	Mnemonic          string    `json:"mnemonic,omitempty"`
	EncryptedMnemonic string    `json:"encrypted_mnemonic,omitempty"`
	Address           string    `json:"address"`
	RawAddrHex        string    `json:"raw_addr_hex"`
	Time              time.Time `json:"time"`
	Exists            bool      `json:"exists"`
}

func matchKey(address string) []byte {
	return []byte("match:" + address)
}

// SaveMatch 保存匹配记录（以 address 为唯一键，幂等）
func (tm *TaskManager) SaveMatch(r *MatchRecord) error {
	data, err := json.Marshal(r)
	if err != nil {
		return err
	}
	return tm.db.Put(matchKey(r.Address), data, nil)
}

// UpdateMatchExists 更新匹配记录的账户存在状态
func (tm *TaskManager) UpdateMatchExists(address string, exists bool) error {
	key := matchKey(address)
	data, err := tm.db.Get(key, nil)
	if err != nil {
		return err
	}
	var r MatchRecord
	if err := json.Unmarshal(data, &r); err != nil {
		return err
	}
	r.Exists = exists
	updated, err := json.Marshal(&r)
	if err != nil {
		return err
	}
	return tm.db.Put(key, updated, nil)
}

// LoadAllMatches 从 LevelDB 加载所有匹配记录
func (tm *TaskManager) LoadAllMatches() ([]*MatchRecord, error) {
	prefix := []byte("match:")
	var records []*MatchRecord
	iter := tm.db.NewIterator(util.BytesPrefix(prefix), nil)
	defer iter.Release()
	for iter.Next() {
		var r MatchRecord
		if err := json.Unmarshal(iter.Value(), &r); err != nil {
			continue
		}
		records = append(records, &r)
	}
	return records, iter.Error()
}
