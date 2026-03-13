package scheduler

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// Job 任务作业
type Job struct {
	ID          string    `json:"id"`
	Name        string    `json:"name"`
	Mnemonic    string    `json:"mnemonic"`
	BatchSize   int       `json:"batch_size"`
	Status      string    `json:"status"` // pending, running, paused, completed
	Total       int64     `json:"total"`
	Completed   int64     `json:"completed"`
	Matches     int64     `json:"matches"`
	CreatedAt   time.Time `json:"created_at"`
	StartedAt   *time.Time `json:"started_at"`
	CompletedAt *time.Time `json:"completed_at"`
}

// State 持久化状态
type State struct {
	Jobs      map[string]*Job `json:"jobs"`
	UpdatedAt time.Time       `json:"updated_at"`
}

// TaskManager 任务管理器
type TaskManager struct {
	jobs   map[string]*Job
	mu     sync.RWMutex
	file   string
	autoID int
}

// NewTaskManager 创建任务管理器
func NewTaskManager(dataDir string) (*TaskManager, error) {
	if err := os.MkdirAll(dataDir, 0755); err != nil {
		return nil, err
	}

	tm := &TaskManager{
		jobs: make(map[string]*Job),
		file: filepath.Join(dataDir, "state.json"),
	}

	// 尝试恢复状态
	if err := tm.load(); err != nil {
		if !os.IsNotExist(err) {
			return nil, fmt.Errorf("failed to load state: %w", err)
		}
	}

	return tm, nil
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
	tm.save()

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

	if j, exists := tm.jobs[job.ID]; exists {
		*j = *job
		tm.save()
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
	tm.save()
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
	tm.save()
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
	tm.save()
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
	tm.save()
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
		tm.save()
	}
}

// IncrementCompleted 增加完成计数
func (tm *TaskManager) IncrementCompleted(id string, matches int) {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	if job, exists := tm.jobs[id]; exists {
		job.Completed++
		job.Matches += int64(matches)
		tm.save()
	}
}

// SetTotal 设置总数
func (tm *TaskManager) SetTotal(id string, total int64) {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	if job, exists := tm.jobs[id]; exists {
		job.Total = total
		tm.save()
	}
}

// SetCompleted 设置已完成数（由 worker 提交结果时更新）
func (tm *TaskManager) SetCompleted(id string, completed int64) {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	if job, exists := tm.jobs[id]; exists {
		job.Completed = completed
		tm.save()
	}
}

// load 加载状态
func (tm *TaskManager) load() error {
	data, err := os.ReadFile(tm.file)
	if err != nil {
		return err
	}

	var state State
	if err := json.Unmarshal(data, &state); err != nil {
		return err
	}

	tm.jobs = state.Jobs

	// 恢复 autoID
	for id := range tm.jobs {
		var num int
		if _, err := fmt.Sscanf(id, "job-%d", &num); err == nil && num > tm.autoID {
			tm.autoID = num
		}
	}

	return nil
}

// save 保存状态
func (tm *TaskManager) save() {
	state := State{
		Jobs:      tm.jobs,
		UpdatedAt: time.Now(),
	}

	data, err := json.MarshalIndent(state, "", "  ")
	if err != nil {
		return
	}

	os.WriteFile(tm.file, data, 0644)
}
