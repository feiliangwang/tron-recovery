package main

import (
	"boon/internal/bloom"
	"boon/internal/compute"
	"boon/internal/mnemonic"
	"boon/internal/protocol"
	"boon/internal/scheduler"
	"encoding/hex"
	"encoding/json"
	"embed"
	"flag"
	"fmt"
	"log"
	"net/http"
	"strings"
	"sync"
	"time"
)

//go:embed static
var staticFS embed.FS

var (
	dataDir   = flag.String("data", "./data", "数据目录")
	port      = flag.Int("port", 8080, "HTTP服务端口")
	bloomFile = flag.String("bloom", "", "Bloom过滤器文件（gob格式，全局）")
)

// Server 服务器
type Server struct {
	taskManager *scheduler.TaskManager
	compute     compute.SeedComputer
	bloomFilter *bloom.Filter // 全局Bloom过滤器

	activeJobs   map[string]*JobRunner
	activeJobsMu sync.RWMutex

	matches   []Match
	matchesMu sync.Mutex

	workers   map[string]*WorkerInfo
	workersMu sync.RWMutex
}

// JobRunner 任务运行器
type JobRunner struct {
	Job       *scheduler.Job
	StopCh    chan struct{}
	Running   bool
	TaskQueue chan *protocol.Task
}

// WorkerInfo Worker信息
type WorkerInfo struct {
	ID        string
	LastSeen  time.Time
	TasksDone int
}

// Match 匹配结果
type Match struct {
	JobID    string    `json:"job_id"`
	Mnemonic string    `json:"mnemonic"`
	Address  string    `json:"address"`
	Time     time.Time `json:"time"`
}

func main() {
	flag.Parse()

	// 创建任务管理器
	tm, err := scheduler.NewTaskManager(*dataDir)
	if err != nil {
		log.Fatalf("创建任务管理器失败: %v", err)
	}

	// 加载Bloom过滤器（全局）
	var bf *bloom.Filter
	if *bloomFile != "" {
		log.Printf("加载Bloom过滤器: %s", *bloomFile)
		bf, err = bloom.LoadFromFile(*bloomFile)
		if err != nil {
			log.Fatalf("加载Bloom过滤器失败: %v", err)
		}
		log.Println("Bloom过滤器加载完成")
	}

	// 创建服务器
	server := &Server{
		taskManager: tm,
		compute:     compute.NewCPUComputer(8),
		bloomFilter: bf,
		activeJobs:  make(map[string]*JobRunner),
		workers:     make(map[string]*WorkerInfo),
	}

	// 恢复运行中的任务
	server.restoreJobs()

	// 启动Worker清理协程
	go server.cleanWorkers()

	// 路由
	http.HandleFunc("/", server.handleIndex)
	http.HandleFunc("/api/jobs", server.handleJobs)
	http.HandleFunc("/api/jobs/", server.handleJobAction)
	http.HandleFunc("/api/task/fetch", server.handleFetchTask)
	http.HandleFunc("/api/task/submit", server.handleSubmitResult)
	http.HandleFunc("/api/matches", server.handleMatches)
	http.HandleFunc("/api/workers", server.handleWorkers)
	http.HandleFunc("/api/stats", server.handleStats)
	http.HandleFunc("/api/bloom", server.handleBloomInfo)

	addr := fmt.Sprintf(":%d", *port)
	log.Printf("服务器启动: http://localhost%s", addr)
	log.Fatal(http.ListenAndServe(addr, nil))
}

// handleIndex 首页
func (s *Server) handleIndex(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}

	data, err := staticFS.ReadFile("static/index.html")
	if err != nil {
		http.Error(w, "index.html not found", 500)
		return
	}

	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Write(data)
}

// handleJobs 任务列表/创建
func (s *Server) handleJobs(w http.ResponseWriter, r *http.Request) {
	if r.Method == "GET" {
		jobs := s.taskManager.ListJobs()
		json.NewEncoder(w).Encode(map[string]interface{}{
			"jobs":         jobs,
			"bloom_loaded": s.bloomFilter != nil,
		})
		return
	}

	if r.Method == "POST" {
		var req struct {
			Name      string `json:"name"`
			Mnemonic  string `json:"mnemonic"`
			BatchSize int    `json:"batch_size"`
		}

		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), 400)
			return
		}

		if req.BatchSize <= 0 {
			req.BatchSize = 1000
		}

		job := s.taskManager.CreateJob(req.Name, req.Mnemonic, req.BatchSize)
		json.NewEncoder(w).Encode(job)
		return
	}

	http.Error(w, "Method not allowed", 405)
}

// handleJobAction 任务操作
func (s *Server) handleJobAction(w http.ResponseWriter, r *http.Request) {
	path := r.URL.Path
	parts := strings.Split(path, "/")
	if len(parts) < 4 {
		http.Error(w, "Invalid path", 400)
		return
	}

	jobID := parts[3]
	action := ""
	if len(parts) > 4 {
		action = parts[4]
	}

	switch r.Method {
	case "GET":
		job := s.taskManager.GetJob(jobID)
		if job == nil {
			http.Error(w, "Job not found", 404)
			return
		}
		json.NewEncoder(w).Encode(job)

	case "DELETE":
		s.stopJob(jobID)
		s.taskManager.DeleteJob(jobID)
		w.WriteHeader(200)

	case "POST":
		switch action {
		case "start":
			s.startJob(jobID)
			w.WriteHeader(200)
		case "pause":
			s.pauseJob(jobID)
			w.WriteHeader(200)
		case "resume":
			s.resumeJob(jobID)
			w.WriteHeader(200)
		default:
			http.Error(w, "Unknown action", 400)
		}

	default:
		http.Error(w, "Method not allowed", 405)
	}
}

// startJob 启动任务
func (s *Server) startJob(jobID string) {
	job := s.taskManager.GetJob(jobID)
	if job == nil {
		return
	}

	s.activeJobsMu.Lock()
	defer s.activeJobsMu.Unlock()

	if _, exists := s.activeJobs[jobID]; exists {
		return
	}

	runner := &JobRunner{
		Job:       job,
		StopCh:    make(chan struct{}),
		TaskQueue: make(chan *protocol.Task, 100),
	}

	s.activeJobs[jobID] = runner
	s.taskManager.StartJob(jobID)
	go s.runEnumerator(runner)
}

// pauseJob 暂停任务
func (s *Server) pauseJob(jobID string) {
	s.activeJobsMu.Lock()
	defer s.activeJobsMu.Unlock()

	if runner, exists := s.activeJobs[jobID]; exists {
		runner.Running = false
	}
	s.taskManager.PauseJob(jobID)
}

// resumeJob 恢复任务
func (s *Server) resumeJob(jobID string) {
	s.activeJobsMu.Lock()
	defer s.activeJobsMu.Unlock()

	if runner, exists := s.activeJobs[jobID]; exists {
		runner.Running = true
	}
	s.taskManager.ResumeJob(jobID)
}

// stopJob 停止任务
func (s *Server) stopJob(jobID string) {
	s.activeJobsMu.Lock()
	defer s.activeJobsMu.Unlock()

	if runner, exists := s.activeJobs[jobID]; exists {
		close(runner.StopCh)
		delete(s.activeJobs, jobID)
	}
}

// runEnumerator 运行枚举器
func (s *Server) runEnumerator(runner *JobRunner) {
	words := strings.Fields(runner.Job.Mnemonic)
	enum := mnemonic.NewEnumerator(words, runner.Job.BatchSize)
	batchChan := enum.BatchEnumerate()

	runner.Running = true

	taskID := 0
	for batch := range batchChan {
		select {
		case <-runner.StopCh:
			return
		default:
		}

		if !runner.Running {
			for !runner.Running {
				time.Sleep(100 * time.Millisecond)
				select {
				case <-runner.StopCh:
					return
				default:
				}
			}
		}

		taskID++
		task := &protocol.Task{
			ID:        taskID,
			Mnemonics: batch,
		}
		runner.TaskQueue <- task
	}

	s.taskManager.CompleteJob(runner.Job.ID)
}

// handleFetchTask 获取任务
func (s *Server) handleFetchTask(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "Method not allowed", 405)
		return
	}

	var req struct {
		WorkerID string `json:"worker_id"`
	}
	json.NewDecoder(r.Body).Decode(&req)

	s.workersMu.Lock()
	s.workers[req.WorkerID] = &WorkerInfo{
		ID:       req.WorkerID,
		LastSeen: time.Now(),
	}
	s.workersMu.Unlock()

	s.activeJobsMu.RLock()
	defer s.activeJobsMu.RUnlock()

	for _, runner := range s.activeJobs {
		if !runner.Running {
			continue
		}

		select {
		case task := <-runner.TaskQueue:
			json.NewEncoder(w).Encode(protocol.TaskResponse{
				Task:  task,
				Count: len(runner.TaskQueue),
			})
			return
		default:
		}
	}

	w.WriteHeader(http.StatusNoContent)
}

// handleSubmitResult 提交结果
func (s *Server) handleSubmitResult(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "Method not allowed", 405)
		return
	}

	var req struct {
		WorkerID  string     `json:"worker_id"`
		TaskID    int        `json:"task_id"`
		Addresses []string   `json:"addresses"`
		Mnemonics [][]string `json:"mnemonics"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), 400)
		return
	}

	s.workersMu.Lock()
	if worker, exists := s.workers[req.WorkerID]; exists {
		worker.TasksDone++
		worker.LastSeen = time.Now()
	}
	s.workersMu.Unlock()

	matches := 0
	for i, addrHex := range req.Addresses {
		if addrHex == "" || len(req.Mnemonics) <= i {
			continue
		}

		addr, err := hex.DecodeString(addrHex)
		if err != nil {
			continue
		}

		// 使用全局Bloom过滤器检查
		if s.bloomFilter != nil && s.bloomFilter.Contains(addr) {
			mnemonicStr := strings.Join(req.Mnemonics[i], " ")

			s.matchesMu.Lock()
			s.matches = append(s.matches, Match{
				Mnemonic: mnemonicStr,
				Address:  addrHex,
				Time:     time.Now(),
			})
			s.matchesMu.Unlock()

			matches++

			log.Printf("========== 找到匹配 ==========")
			log.Printf("Worker: %s", req.WorkerID)
			log.Printf("助记词: %s", mnemonicStr)
			log.Printf("地址: %s", addrHex)
			log.Printf("==============================")
		}
	}

	json.NewEncoder(w).Encode(protocol.ResultResponse{
		Success: true,
		Matches: matches,
	})
}

// handleMatches 匹配结果
func (s *Server) handleMatches(w http.ResponseWriter, r *http.Request) {
	s.matchesMu.Lock()
	matches := s.matches
	s.matchesMu.Unlock()

	json.NewEncoder(w).Encode(map[string]interface{}{"matches": matches})
}

// handleWorkers Worker列表
func (s *Server) handleWorkers(w http.ResponseWriter, r *http.Request) {
	s.workersMu.RLock()
	workers := make([]*WorkerInfo, 0, len(s.workers))
	for _, w := range s.workers {
		workers = append(workers, w)
	}
	s.workersMu.RUnlock()

	json.NewEncoder(w).Encode(map[string]interface{}{"workers": workers})
}

// handleStats 统计信息
func (s *Server) handleStats(w http.ResponseWriter, r *http.Request) {
	stats := map[string]interface{}{
		"jobs":         len(s.taskManager.ListJobs()),
		"workers":      len(s.workers),
		"matches":      len(s.matches),
		"bloom_loaded": s.bloomFilter != nil,
	}
	json.NewEncoder(w).Encode(stats)
}

// handleBloomInfo Bloom过滤器信息
func (s *Server) handleBloomInfo(w http.ResponseWriter, r *http.Request) {
	info := map[string]interface{}{
		"loaded": s.bloomFilter != nil,
		"file":   *bloomFile,
	}
	json.NewEncoder(w).Encode(info)
}

// restoreJobs 恢复任务
func (s *Server) restoreJobs() {
	jobs := s.taskManager.ListJobs()
	for _, job := range jobs {
		if job.Status == "running" {
			job.Status = "paused"
			s.taskManager.UpdateJob(job)
			log.Printf("恢复任务: %s (状态: paused)", job.ID)
		}
	}
}

// cleanWorkers 清理过期worker
func (s *Server) cleanWorkers() {
	ticker := time.NewTicker(30 * time.Second)
	for range ticker.C {
		s.workersMu.Lock()
		for id, w := range s.workers {
			if time.Since(w.LastSeen) > 60*time.Second {
				delete(s.workers, id)
			}
		}
		s.workersMu.Unlock()
	}
}
