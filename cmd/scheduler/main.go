package main

import (
	"bytes"
	"embed"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"boon/internal/bip44"
	"boon/internal/mnemonic"
	"boon/internal/protocol"
	"boon/internal/scheduler"
	"boon/internal/tron"
)

//go:embed static
var staticFS embed.FS

var (
	dataDir = flag.String("data", "./data", "数据目录")
	port    = flag.Int("port", 8080, "HTTP服务端口")
)

// pendingTaskRecord 记录已分发但尚未提交结果的任务
type pendingTaskRecord struct {
	jobID    string
	startIdx int64
	endIdx   int64
}

// Server 服务器
type Server struct {
	taskManager *scheduler.TaskManager
	matchesFile string // 匹配结果持久化文件路径

	jobs   map[string]*CompactJobRunner
	jobsMu sync.RWMutex

	pendingTasks   map[int64]pendingTaskRecord // TaskID -> 记录
	pendingTasksMu sync.Mutex

	matches   []*Match
	matchesMu sync.Mutex

	workers   map[string]*WorkerInfo
	workersMu sync.RWMutex

	tronClient *tron.Client // TRON HTTP API 客户端，用于确认账户状态
}

// CompactJobRunner 紧凑任务运行器
type CompactJobRunner struct {
	Job          *scheduler.Job
	Template     *protocol.TaskTemplate
	CurrentIdx   int64 // 已分发的索引上边界
	CompletedIdx int64 // Worker 已提交结果的累计数量（用于速度/ETA 显示）
	CommittedIdx int64 // 安全重启点：所有低于此点的索引均已完成（飞行中任务的最小 StartIdx）
	TotalIdx     int64
	BatchSize    int64
	StopCh       chan struct{}
	Running      bool

	ActiveSince       time.Time // 最近一次激活时间（启动/恢复）
	CompletedAtActive int64     // 激活时的 CompletedIdx（用于计算当前阶段速度）
}

// JobView 任务的完整视图（含实时计算字段）
type JobView struct {
	ID        string     `json:"id"`
	Name      string     `json:"name"`
	Mnemonic  string     `json:"mnemonic"`
	Status    string     `json:"status"`
	Total     int64      `json:"total"`
	Completed int64      `json:"completed"`
	Matches   int64      `json:"matches"`
	CreatedAt time.Time  `json:"created_at"`
	StartedAt *time.Time `json:"started_at"`

	// 实时计算字段
	ElapsedSeconds int64 `json:"elapsed_seconds"` // 自首次启动起累计壁钟秒数
	Speed          int64 `json:"speed"`           // 当前阶段枚举速度（本次激活以来）
	ETASeconds     int64 `json:"eta_seconds"`     // 预计剩余秒数，-1=无法估算
}

// WorkerInfo Worker信息
type WorkerInfo struct {
	ID        string    `json:"id"`
	LastSeen  time.Time `json:"last_seen"`
	TasksDone int64     `json:"tasks_done"`
	Speed     int64     `json:"speed"` // 枚举速度（indices/s），由Worker握手时上报
}

const (
	// targetBatchSeconds 目标每批任务的计算时长（秒）
	targetBatchSeconds = 30
	// minBatchSize 最小批次大小（冷启动或速度未知时使用）
	minBatchSize = 1000
	// maxBatchSize 最大批次大小（防止单次分配过多）
	maxBatchSize = 2_000_000
)

// Match 匹配结果
type Match struct {
	JobID      string    `json:"job_id"`
	Mnemonic   string    `json:"mnemonic"`
	Address    string    `json:"address"`      // TRON Base58Check 地址
	RawAddrHex string    `json:"raw_addr_hex"` // 20字节地址的 hex 编码，用于重启后恢复查询能力
	Time       time.Time `json:"time"`
	Confirmed  *bool     `json:"confirmed"`   // nil=待确认, true=已激活, false=未激活
	CreateTime int64     `json:"create_time"` // 账户创建时间（毫秒时间戳）
	ConfirmErr bool      `json:"confirm_err"` // true=HTTP查询失败，非账户未激活
	rawAddr    []byte    // 运行时原始20字节地址（从 RawAddrHex 恢复）
}

func main() {
	flag.Parse()

	// 创建任务管理器
	tm, err := scheduler.NewTaskManager(*dataDir)
	if err != nil {
		log.Fatalf("创建任务管理器失败: %v", err)
	}

	// 创建服务器
	server := &Server{
		taskManager:  tm,
		matchesFile:  filepath.Join(*dataDir, "matches.json"),
		jobs:         make(map[string]*CompactJobRunner),
		pendingTasks: make(map[int64]pendingTaskRecord),
		workers:      make(map[string]*WorkerInfo),
	}

	// 初始化 TRON HTTP API 客户端
	if tc, err := tron.NewClient(tron.DefaultEndpoint); err != nil {
		log.Printf("TRON HTTP 客户端初始化失败（确认功能不可用）: %v", err)
	} else {
		server.tronClient = tc
		log.Printf("TRON HTTP 客户端已就绪: %s", tron.DefaultEndpoint)
	}

	// 恢复任务
	server.restoreJobs()

	// 加载持久化的匹配结果
	server.loadMatches()

	// Worker清理
	go server.cleanWorkers()

	// 后台重试确认失败的 match
	go server.retryFailedConfirms()

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
		// 收集 runner 实时状态
		s.jobsMu.RLock()
		type runnerSnap struct {
			completedIdx      int64
			totalIdx          int64
			running           bool
			activeSince       time.Time
			completedAtActive int64
		}
		snaps := make(map[string]runnerSnap, len(s.jobs))
		for id, r := range s.jobs {
			snaps[id] = runnerSnap{
				completedIdx:      r.CompletedIdx,
				totalIdx:          r.TotalIdx,
				running:           r.Running,
				activeSince:       r.ActiveSince,
				completedAtActive: r.CompletedAtActive,
			}
		}
		s.jobsMu.RUnlock()

		jobs := s.taskManager.ListJobs()
		views := make([]JobView, 0, len(jobs))
		now := time.Now()
		for _, job := range jobs {
			v := JobView{
				ID:         job.ID,
				Name:       job.Name,
				Mnemonic:   job.Mnemonic,
				Status:     job.Status,
				Total:      job.Total,
				Completed:  job.Completed,
				Matches:    job.Matches,
				CreatedAt:  job.CreatedAt,
				StartedAt:  job.StartedAt,
				ETASeconds: -1,
			}

			if snap, ok := snaps[job.ID]; ok {
				v.Completed = snap.completedIdx

				// 运行时长：自首次启动的壁钟时间
				if job.StartedAt != nil {
					v.ElapsedSeconds = int64(now.Sub(*job.StartedAt).Seconds())
				}

				// 速度 & ETA：仅在运行中且激活时间 > 1s 时计算
				if snap.running && !snap.activeSince.IsZero() {
					activeDur := now.Sub(snap.activeSince)
					if activeDur > time.Second {
						done := snap.completedIdx - snap.completedAtActive
						if done > 0 {
							v.Speed = int64(float64(done) / activeDur.Seconds())
							remaining := snap.totalIdx - snap.completedIdx
							if v.Speed > 0 {
								v.ETASeconds = remaining / v.Speed
							}
						}
					}
				}
			} else if job.StartedAt != nil {
				// 任务已完成或暂停，runner 已移除
				v.ElapsedSeconds = int64(now.Sub(*job.StartedAt).Seconds())
			}

			views = append(views, v)
		}

		json.NewEncoder(w).Encode(map[string]interface{}{"jobs": views})
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
		Job:               job,
		Template:          template,
		CurrentIdx:        job.Completed,
		CompletedIdx:      0,
		CommittedIdx:      job.Completed,
		TotalIdx:          total,
		BatchSize:         int64(job.BatchSize),
		StopCh:            make(chan struct{}),
		Running:           true,
		ActiveSince:       time.Now(),
		CompletedAtActive: 0,
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

	// job 参数是数字 ID，需要在 runners 中按 template.JobID 匹配
	var numericJobID int64
	fmt.Sscanf(jobIDStr, "%d", &numericJobID)

	s.jobsMu.RLock()
	var runner *CompactJobRunner
	for _, r := range s.jobs {
		if r.Template.JobID == numericJobID {
			runner = r
			break
		}
	}
	s.jobsMu.RUnlock()

	if runner == nil {
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
	if 1+workerIDLen+8 > len(body) {
		http.Error(w, "body too short", 400)
		return
	}
	workerID := string(body[1 : 1+workerIDLen])

	// 读取 worker 上报的枚举速度（indices/s）
	var speed int64
	binary.Read(bytes.NewReader(body[1+workerIDLen:]), binary.BigEndian, &speed)

	// 根据速度动态计算本次分配的枚举空间：speed × targetBatchSeconds
	batchSize := speed * targetBatchSeconds
	if batchSize < minBatchSize {
		batchSize = minBatchSize
	}
	if batchSize > maxBatchSize {
		batchSize = maxBatchSize
	}

	// 更新Worker（记录速度）
	s.workersMu.Lock()
	if info, exists := s.workers[workerID]; exists {
		info.LastSeen = time.Now()
		info.Speed = speed
	} else {
		s.workers[workerID] = &WorkerInfo{
			ID:       workerID,
			LastSeen: time.Now(),
			Speed:    speed,
		}
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

		// 记录待确认任务（含区间信息，用于计算连续完成前沿）
		s.pendingTasksMu.Lock()
		s.pendingTasks[task.TaskID] = pendingTaskRecord{jobID: jobID, startIdx: start, endIdx: end}
		s.pendingTasksMu.Unlock()

		w.Header().Set("Content-Type", "application/octet-stream")
		w.Write(task.Encode())

		log.Printf("[Task] %s <- job=%s range=[%d,%d) batch=%d speed=%d/s",
			workerID, jobID, start, end, end-start, speed)
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

	// 更新 CompletedIdx 和 CommittedIdx
	s.pendingTasksMu.Lock()
	rec, found := s.pendingTasks[result.TaskID]
	if found {
		delete(s.pendingTasks, result.TaskID)
	}
	// 计算此 job 的连续完成前沿：飞行中任务的最小 StartIdx
	// 若无飞行中任务，则安全点 = CurrentIdx
	committedByJob := make(map[string]int64)
	for _, r := range s.pendingTasks {
		if cur, ok := committedByJob[r.jobID]; !ok || r.startIdx < cur {
			committedByJob[r.jobID] = r.startIdx
		}
	}
	s.pendingTasksMu.Unlock()

	if found {
		s.jobsMu.Lock()
		for jobID, runner := range s.jobs {
			if jobID == rec.jobID {
				runner.CompletedIdx += rec.endIdx - rec.startIdx
				// 安全重启点：若本 job 还有飞行中任务，取最小 StartIdx；否则取 CurrentIdx
				if minPending, ok := committedByJob[jobID]; ok {
					runner.CommittedIdx = minPending
				} else {
					runner.CommittedIdx = runner.CurrentIdx
				}
				s.taskManager.SetCompleted(jobID, runner.CommittedIdx)

				// 检查是否全部完成
				if runner.CommittedIdx >= runner.TotalIdx {
					runner.Running = false
					delete(s.jobs, jobID)
					s.taskManager.CompleteJob(jobID)
					log.Printf("任务完成: %s 总计=%d", jobID, runner.TotalIdx)
				}
				break
			}
		}
		s.jobsMu.Unlock()
	}

	// 处理匹配
	for _, match := range result.Matches {
		mnemonicStr := s.indexToMnemonic(result.TaskID, match.Index)
		tronAddr := bip44.GetTronAddress(match.Address)

		m := &Match{
			Mnemonic:   mnemonicStr,
			Address:    tronAddr,
			RawAddrHex: hex.EncodeToString(match.Address),
			Time:       time.Now(),
			rawAddr:    match.Address,
		}

		s.matchesMu.Lock()
		s.matches = append(s.matches, m)
		s.saveMatchesLocked()
		s.matchesMu.Unlock()

		log.Printf("========== 匹配 ==========")
		log.Printf("Worker: %s", workerID)
		log.Printf("助记词: %s", mnemonicStr)
		log.Printf("地址: %s", tronAddr)
		log.Printf("==========================")

		// 后台异步确认账户状态
		go s.confirmMatch(m)
	}

	w.WriteHeader(http.StatusOK)
}

// indexToMnemonic 从索引还原助记词
func (s *Server) indexToMnemonic(taskID int64, idx int64) string {
	s.jobsMu.RLock()
	defer s.jobsMu.RUnlock()

	for _, runner := range s.jobs {
		words := make([]string, len(runner.Template.Words))
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
		totalCompleted += runner.CompletedIdx
	}
	s.jobsMu.RUnlock()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"jobs":      len(s.taskManager.ListJobs()),
		"workers":   len(s.workers),
		"matches":   len(s.matches),
		"completed": totalCompleted,
	})
}

// handleBloomInfo Bloom信息（已废弃，保留兼容）
func (s *Server) handleBloomInfo(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"loaded": false,
		"note":   "Bloom filter is now loaded by worker locally",
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
		runner.ActiveSince = time.Now()
		runner.CompletedAtActive = runner.CompletedIdx
	}
	s.taskManager.ResumeJob(jobID)
}


// restoreJobs 恢复任务：重建 runner 并保持暂停状态
func (s *Server) restoreJobs() {
	jobs := s.taskManager.ListJobs()
	for _, job := range jobs {
		if job.Status != "running" && job.Status != "paused" {
			continue
		}

		// 将 running 标记为 paused（未实际运行）
		if job.Status == "running" {
			job.Status = "paused"
			s.taskManager.UpdateJob(job)
		}

		// 重建 runner，恢复到已保存的完成进度
		words := strings.Fields(job.Mnemonic)
		unknownPos := make([]int, 0)
		for i, w := range words {
			if w == "?" {
				unknownPos = append(unknownPos, i)
			}
		}

		wordCount := int64(len(mnemonic.WordList))
		total := job.Total
		if total == 0 {
			total = int64(1)
			for range unknownPos {
				total *= wordCount
			}
		}

		template := &protocol.TaskTemplate{
			JobID:      extractJobNum(job.ID),
			Words:      words,
			UnknownPos: unknownPos,
		}

		runner := &CompactJobRunner{
			Job:               job,
			Template:          template,
			CurrentIdx:        job.Completed, // 从安全前沿重新分发
			CompletedIdx:      0,
			CommittedIdx:      job.Completed,
			TotalIdx:          total,
			BatchSize:         int64(job.BatchSize),
			StopCh:            make(chan struct{}),
			Running:           false, // 暂停状态，等待用户恢复
			ActiveSince:       time.Time{},
			CompletedAtActive: 0,
		}

		s.jobsMu.Lock()
		s.jobs[job.ID] = runner
		s.jobsMu.Unlock()

		log.Printf("任务已恢复（暂停）: %s 进度=%d/%d", job.ID, job.Completed, total)
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

// saveMatchesLocked 将 matches 写入磁盘（调用前须持有 matchesMu）
func (s *Server) saveMatchesLocked() {
	if s.matchesFile == "" {
		return
	}
	data, err := json.MarshalIndent(s.matches, "", "  ")
	if err != nil {
		log.Printf("[Matches] 序列化失败: %v", err)
		return
	}
	if err := os.WriteFile(s.matchesFile, data, 0644); err != nil {
		log.Printf("[Matches] 写入失败: %v", err)
	}
}

// loadMatches 从磁盘恢复 matches，并还原 rawAddr
func (s *Server) loadMatches() {
	if s.matchesFile == "" {
		return
	}
	data, err := os.ReadFile(s.matchesFile)
	if err != nil {
		if !os.IsNotExist(err) {
			log.Printf("[Matches] 读取失败: %v", err)
		}
		return
	}

	var matches []*Match
	if err := json.Unmarshal(data, &matches); err != nil {
		log.Printf("[Matches] 解析失败: %v", err)
		return
	}

	// 从 RawAddrHex 恢复 rawAddr，对仍需重试的重新标记 ConfirmErr
	for _, m := range matches {
		if m.RawAddrHex != "" {
			if b, err := hex.DecodeString(m.RawAddrHex); err == nil {
				m.rawAddr = b
			}
		}
	}

	s.matchesMu.Lock()
	s.matches = matches
	s.matchesMu.Unlock()

	log.Printf("[Matches] 已加载 %d 条匹配记录", len(matches))
}

// extractJobNum 提取job数字
func extractJobNum(jobID string) int64 {
	var num int64
	fmt.Sscanf(jobID, "job-%d", &num)
	return num
}

// confirmMatch 后台调用 TRON HTTP API 确认账户是否已激活（单次尝试，失败后由 retryFailedConfirms 重试）
func (s *Server) confirmMatch(m *Match) {
	if s.tronClient == nil {
		s.matchesMu.Lock()
		m.ConfirmErr = true
		s.matchesMu.Unlock()
		return
	}

	activated, createTime, err := s.tronClient.IsActivated(m.rawAddr)
	if err != nil {
		log.Printf("[Confirm] HTTP 查询失败 %s: %v", m.Address, err)
		s.matchesMu.Lock()
		m.ConfirmErr = true
		s.matchesMu.Unlock()
		return
	}

	s.matchesMu.Lock()
	m.Confirmed = &activated
	m.CreateTime = createTime
	m.ConfirmErr = false
	s.saveMatchesLocked()
	s.matchesMu.Unlock()

	if activated {
		log.Printf("[Confirm] ✅ 账户已激活 %s create_time=%d", m.Address, createTime)
	} else {
		log.Printf("[Confirm] ❌ 账户未激活 %s", m.Address)
	}
}

// retryFailedConfirms 定时重试所有确认失败的 match，指数退避，直到成功为止
func (s *Server) retryFailedConfirms() {
	// 退避间隔序列：30s, 1m, 2m, 5m, 10m，之后固定 10m 轮询
	backoff := []time.Duration{
		30 * time.Second,
		1 * time.Minute,
		2 * time.Minute,
		5 * time.Minute,
		10 * time.Minute,
	}

	type retryState struct {
		attempt int
		nextAt  time.Time
	}
	states := make(map[*Match]*retryState)

	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	for range ticker.C {
		if s.tronClient == nil {
			continue
		}

		now := time.Now()

		s.matchesMu.Lock()
		var toRetry []*Match
		for _, m := range s.matches {
			if !m.ConfirmErr {
				continue
			}
			st, ok := states[m]
			if !ok {
				st = &retryState{attempt: 0, nextAt: now}
				states[m] = st
			}
			if now.Before(st.nextAt) {
				continue
			}
			toRetry = append(toRetry, m)
		}
		s.matchesMu.Unlock()

		for _, m := range toRetry {
			st := states[m]
			log.Printf("[Confirm] 重试确认（第 %d 次）: %s", st.attempt+1, m.Address)

			activated, createTime, err := s.tronClient.IsActivated(m.rawAddr)
			if err != nil {
				log.Printf("[Confirm] 重试失败（第 %d 次）%s: %v", st.attempt+1, m.Address, err)
				st.attempt++
				idx := st.attempt
				if idx >= len(backoff) {
					idx = len(backoff) - 1
				}
				st.nextAt = now.Add(backoff[idx])
				continue
			}

			s.matchesMu.Lock()
			m.Confirmed = &activated
			m.CreateTime = createTime
			m.ConfirmErr = false
			s.saveMatchesLocked()
			s.matchesMu.Unlock()

			delete(states, m)

			if activated {
				log.Printf("[Confirm] ✅ 重试成功，账户已激活 %s create_time=%d", m.Address, createTime)
			} else {
				log.Printf("[Confirm] ❌ 重试成功，账户未激活 %s", m.Address)
			}
		}
	}
}
