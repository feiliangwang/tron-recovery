package main

import (
	"bytes"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"embed"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"sync"
	"time"

	"boon/internal/bloom"
	"boon/internal/mnemonic"
	"boon/internal/protocol"
	"boon/internal/scheduler"
)

//go:embed static
var staticFS embed.FS

var (
	dataDir   = flag.String("data", "./data", "数据目录")
	port      = flag.Int("port", 8080, "HTTP服务端口")
	bloomFile = flag.String("bloom", "", "Bloom过滤器文件（gob格式）")
)

// Server 服务器
type Server struct {
	taskManager *scheduler.TaskManager
	bloomFilter *bloom.Filter

	jobs   map[string]*CompactJobRunner
	jobsMu sync.RWMutex

	matches   []Match
	matchesMu sync.Mutex

	workers   map[string]*WorkerInfo
	workersMu sync.RWMutex
}

// CompactJobRunner 紧凑任务运行器
type CompactJobRunner struct {
	Job        *scheduler.Job
	Template   *protocol.TaskTemplate
	CurrentIdx int64
	TotalIdx   int64
	BatchSize  int64
	StopCh     chan struct{}
	Running    bool
}

// WorkerInfo Worker信息
type WorkerInfo struct {
	ID        string    `json:"id"`
	LastSeen  time.Time `json:"last_seen"`
	TasksDone int64     `json:"tasks_done"`
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

	// 加载Bloom过滤器
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
		bloomFilter: bf,
		jobs:        make(map[string]*CompactJobRunner),
		workers:     make(map[string]*WorkerInfo),
	}

	// 恢复任务
	server.restoreJobs()

	// Worker清理
	go server.cleanWorkers()

	// 路由
	http.HandleFunc("/", server.handleIndex)
	http.HandleFunc("/api/jobs", server.handleJobs)
	http.HandleFunc("/api/jobs/", server.handleJobAction)
	http.HandleFunc("/api/template", server.handleTemplate)
	http.HandleFunc("/api/task/fetch", server.handleTaskFetch)
	http.HandleFunc("/api/task/submit", server.handleTaskSubmit)
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
	data, _ := staticFS.ReadFile("static/index.html")
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Write(data)
}

// handleJobs 任务列表/创建
func (s *Server) handleJobs(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

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
		json.NewDecoder(r.Body).Decode(&req)

		if req.BatchSize <= 0 {
			req.BatchSize = 10000
		}

		job := s.taskManager.CreateJob(req.Name, req.Mnemonic, req.BatchSize)
		json.NewEncoder(w).Encode(job)
		return
	}

	http.Error(w, "Method not allowed", 405)
}

// handleJobAction 任务操作
func (s *Server) handleJobAction(w http.ResponseWriter, r *http.Request) {
	parts := strings.Split(r.URL.Path, "/")
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
		w.Header().Set("Content-Type", "application/json")
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

	s.jobsMu.Lock()
	defer s.jobsMu.Unlock()

	if _, exists := s.jobs[jobID]; exists {
		return
	}

	// 解析助记词模板
	words := strings.Fields(job.Mnemonic)
	unknownPos := make([]int, 0)
	for i, w := range words {
		if w == "?" {
			unknownPos = append(unknownPos, i)
		}
	}

	// 计算总组合数
	wordCount := int64(len(mnemonic.WordList))
	total := int64(1)
	for i := 0; i < len(unknownPos); i++ {
		total *= wordCount
	}

	// 创建模板
	template := &protocol.TaskTemplate{
		JobID:      extractJobNum(jobID),
		Words:      words,
		UnknownPos: unknownPos,
	}

	runner := &CompactJobRunner{
		Job:        job,
		Template:   template,
		CurrentIdx: 0,
		TotalIdx:   total,
		BatchSize:  int64(job.BatchSize),
		StopCh:     make(chan struct{}),
		Running:    true,
	}

	s.jobs[jobID] = runner
	s.taskManager.StartJob(jobID)
	s.taskManager.SetTotal(jobID, total)

	log.Printf("任务启动: %s 总组合=%d 未知位置=%v", jobID, total, unknownPos)
}

// handleTemplate 获取任务模板
func (s *Server) handleTemplate(w http.ResponseWriter, r *http.Request) {
	jobIDStr := r.URL.Query().Get("job")
	if jobIDStr == "" {
		http.Error(w, "missing job parameter", 400)
		return
	}

	s.jobsMu.RLock()
	runner, exists := s.jobs[jobIDStr]
	s.jobsMu.RUnlock()

	if !exists {
		w.WriteHeader(http.StatusNoContent)
		return
	}

	data := runner.Template.Encode()
	w.Header().Set("Content-Type", "application/octet-stream")
	w.Write(data)
}

// handleTaskFetch 获取任务（二进制）
func (s *Server) handleTaskFetch(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "Method not allowed", 405)
		return
	}

	body, _ := io.ReadAll(r.Body)
	if len(body) < 1 {
		http.Error(w, "empty body", 400)
		return
	}

	workerIDLen := int(body[0])
	workerID := string(body[1 : 1+workerIDLen])
	var batchSize int64
	binary.Read(bytes.NewReader(body[1+workerIDLen:]), binary.BigEndian, &batchSize)

	// 更新Worker
	s.workersMu.Lock()
	s.workers[workerID] = &WorkerInfo{
		ID:       workerID,
		LastSeen: time.Now(),
	}
	s.workersMu.Unlock()

	// 分配任务
	s.jobsMu.Lock()
	defer s.jobsMu.Unlock()

	for jobID, runner := range s.jobs {
		if !runner.Running {
			continue
		}

		start := runner.CurrentIdx
		if start >= runner.TotalIdx {
			continue
		}

		end := start + batchSize
		if end > runner.TotalIdx {
			end = runner.TotalIdx
		}
		runner.CurrentIdx = end

		task := &protocol.CompactTask{
			TaskID:   time.Now().UnixNano(),
			JobID:    runner.Template.JobID,
			StartIdx: start,
			EndIdx:   end,
		}

		w.Header().Set("Content-Type", "application/octet-stream")
		w.Write(task.Encode())

		log.Printf("[Task] %s <- job=%s range=[%d,%d)", workerID, jobID, start, end)
		return
	}

	w.WriteHeader(http.StatusNoContent)
}

// handleTaskSubmit 提交结果（二进制）
func (s *Server) handleTaskSubmit(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "Method not allowed", 405)
		return
	}

	body, _ := io.ReadAll(r.Body)
	if len(body) < 1 {
		http.Error(w, "empty body", 400)
		return
	}

	workerIDLen := int(body[0])
	workerID := string(body[1 : 1+workerIDLen])
	resultData := body[1+workerIDLen:]

	result, err := protocol.DecodeCompactResult(resultData)
	if err != nil {
		http.Error(w, err.Error(), 400)
		return
	}

	// 更新Worker
	s.workersMu.Lock()
	if w, exists := s.workers[workerID]; exists {
		w.TasksDone++
		w.LastSeen = time.Now()
	}
	s.workersMu.Unlock()

	// 处理匹配
	for _, match := range result.Matches {
		mnemonicStr := s.indexToMnemonic(result.TaskID, match.Index)
		addrHex := hex.EncodeToString(match.Address)

		s.matchesMu.Lock()
		s.matches = append(s.matches, Match{
			Mnemonic: mnemonicStr,
			Address:  addrHex,
			Time:     time.Now(),
		})
		s.matchesMu.Unlock()

		log.Printf("========== 匹配 ==========")
		log.Printf("Worker: %s", workerID)
		log.Printf("助记词: %s", mnemonicStr)
		log.Printf("地址: %s", addrHex)
		log.Printf("==========================")
	}

	w.WriteHeader(http.StatusOK)
}

// indexToMnemonic 从索引还原助记词
func (s *Server) indexToMnemonic(taskID int64, idx int64) string {
	s.jobsMu.RLock()
	defer s.jobsMu.RUnlock()

	for _, runner := range s.jobs {
		words := make([]string, 12)
		copy(words, runner.Template.Words)

		wordCount := int64(len(mnemonic.WordList))
		remaining := idx
		for _, pos := range runner.Template.UnknownPos {
			wordIdx := remaining % wordCount
			remaining /= wordCount
			words[pos] = mnemonic.WordList[wordIdx]
		}

		return strings.Join(words, " ")
	}

	return ""
}

// handleMatches 匹配结果
func (s *Server) handleMatches(w http.ResponseWriter, r *http.Request) {
	s.matchesMu.Lock()
	matches := s.matches
	s.matchesMu.Unlock()
	w.Header().Set("Content-Type", "application/json")
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
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{"workers": workers})
}

// handleStats 统计
func (s *Server) handleStats(w http.ResponseWriter, r *http.Request) {
	s.jobsMu.RLock()
	totalCompleted := int64(0)
	for _, runner := range s.jobs {
		totalCompleted += runner.CurrentIdx
	}
	s.jobsMu.RUnlock()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"jobs":         len(s.taskManager.ListJobs()),
		"workers":      len(s.workers),
		"matches":      len(s.matches),
		"bloom_loaded": s.bloomFilter != nil,
		"completed":    totalCompleted,
	})
}

// handleBloomInfo Bloom信息
func (s *Server) handleBloomInfo(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"loaded": s.bloomFilter != nil,
		"file":   *bloomFile,
	})
}

// stopJob 停止任务
func (s *Server) stopJob(jobID string) {
	s.jobsMu.Lock()
	defer s.jobsMu.Unlock()
	if runner, exists := s.jobs[jobID]; exists {
		close(runner.StopCh)
		delete(s.jobs, jobID)
	}
}

// pauseJob 暂停任务
func (s *Server) pauseJob(jobID string) {
	s.jobsMu.Lock()
	defer s.jobsMu.Unlock()
	if runner, exists := s.jobs[jobID]; exists {
		runner.Running = false
	}
	s.taskManager.PauseJob(jobID)
}

// resumeJob 恢复任务
func (s *Server) resumeJob(jobID string) {
	s.jobsMu.Lock()
	defer s.jobsMu.Unlock()
	if runner, exists := s.jobs[jobID]; exists {
		runner.Running = true
	}
	s.taskManager.ResumeJob(jobID)
}

// restoreJobs 恢复任务
func (s *Server) restoreJobs() {
	jobs := s.taskManager.ListJobs()
	for _, job := range jobs {
		if job.Status == "running" {
			job.Status = "paused"
			s.taskManager.UpdateJob(job)
		}
	}
}

// cleanWorkers 清理Worker
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

// extractJobNum 提取job数字
func extractJobNum(jobID string) int64 {
	var num int64
	fmt.Sscanf(jobID, "job-%d", &num)
	return num
}
