package main

import (
	"bytes"
	"crypto/rand"
	"crypto/rsa"
	"crypto/sha256"
	"crypto/x509"
	"embed"
	"encoding/base64"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"encoding/pem"
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

	"boon/internal/account"
	"boon/internal/bip44"
	"boon/internal/compute"
	"boon/internal/mnemonic"
	"boon/internal/protocol"
	"boon/internal/scheduler"
)

//go:embed static
var staticFS embed.FS

var (
	dataDir   = flag.String("data", "./data", "数据目录")
	port      = flag.Int("port", 8080, "HTTP服务端口")
	accountDb = flag.String("accountdb", "", "账户数据库路径（LevelDB）")
	authFlag  = flag.String("auth", "", "Web UI 认证，格式 username:password（留空则不启用）")
)

// embeddedPublicKey 内嵌 RSA-4096 公钥，助记词加密存储，服务端不持有私钥无法解密
const embeddedPublicKey = `-----BEGIN PUBLIC KEY-----
MIICIjANBgkqhkiG9w0BAQEFAAOCAg8AMIICCgKCAgEAwgvu2PHZ05zKW9TeEBYe
9XlyIoqGCQtBkWFQg9QSQPIP9d1Rn1QL7nKmiFXaSXqItfUtdKGYVvlEOnVPDf7A
Xe/CA6Y/+tKHSMtz2TnTVhkegL5AtZgzgGFalZDLDZNNft/MZw2afe07gZtHVuEB
2y4ipKN52uRHGeKJAzg8DQXJrMTu6BXvOafxlnAyrv7hZlPf2rScvRE4pq2/8qOv
/T3Bccn0jJwYBBAEcEAI+jAdeFgxy5vdy65JvdHTTF9f92nx7whDkV4EqwoiDidD
JgBkIb4hdpqrdzDFBQydZQx/y2giRm1gheqQfZ4cbx1Q/xWw2SYSEHr6zT+ieJHh
RUYBcVAq3hXAc2zoNZUkVjkZemL7PksDmxX81oPO/dJVnNK0Hc6ioZDrxPmosdBQ
72ONwIflVH7FILrPa4pkUf9Z7uJEZifZl/3f0LLnfE3pJPXhUJJX+wafEVHz+Dvf
zDcNr1Ly6OGGV0lN8F/VAIAkpZfzJxtScFmKHwgbA8yAxR2elqxC7VJEzFPF4ccl
USdA2pQsWykxkuL+oO2cXI31C9wUs+4dP8/w1CfmvxaZkXUlPOtoQ9Ig3EpRNKfY
jeJKmRBgS+DWzG/3B8hwHKCKQhcAWXebfzoTOkMWeLzZfGyyVgOGD5oZJ7IVieVo
Ok/kyUpnsHx6Yy9APo4BSAMCAwEAAQ==
-----END PUBLIC KEY-----`

// pendingTaskRecord 记录已分发但尚未提交结果的任务
type pendingTaskRecord struct {
	jobID     string
	startIdx  int64
	endIdx    int64
	fetchedAt time.Time // 任务分发时间，用于计算实际计算耗时
}

// jobSpeedEntry 任务速度滑动窗口条目
type jobSpeedEntry struct {
	t        time.Time
	indices  int64
	duration time.Duration // worker 实际计算耗时（submitTime - fetchTime）
}

// Server 服务器
type Server struct {
	taskManager   *scheduler.TaskManager
	matchesFile   string         // 匹配结果持久化文件路径
	confirmedFile string         // 确认成功的地址持久化文件路径
	pubKey        *rsa.PublicKey // RSA 公钥，非 nil 时启用助记词加密；服务端无私钥，无法解密

	// Web UI 认证
	authUser   string
	authPass   string
	sessions   map[string]time.Time
	sessionsMu sync.Mutex

	jobs   map[string]*CompactJobRunner
	jobsMu sync.RWMutex

	pendingTasks   map[int64]pendingTaskRecord // TaskID -> 记录
	pendingTasksMu sync.Mutex

	matches   []*Match
	matchesMu sync.Mutex

	confirmed   []*Match // 确认成功的地址列表
	confirmedMu sync.Mutex

	workers           map[string]*WorkerInfo
	workersMu         sync.RWMutex
	activeWorkerCount int // 当前在线 worker 数，变化时重置 job 速度窗口

	accountDb      *account.Db          // 账户数据库，用于检查地址是否存在
	seedComputer   compute.SeedComputer // 用于验证提交的地址
	verificationMu sync.Mutex           // 保护 seedComputer
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

	// 1分钟滑动窗口速度（jobsMu 保护）
	speedWindow []jobSpeedEntry

	// 超时缺口队列：记录需要重新分配的索引区间（按 StartIdx 排序）
	gaps []indexRange
}

// indexRange 索引区间
type indexRange struct {
	start int64
	end   int64
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

	// 缺口信息
	GapCount   int64 `json:"gap_count"`   // 缺口数量
	GapIndices int64 `json:"gap_indices"` // 缺口总索引数
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
	// pendingTaskTimeout 飞行中任务超时时间（超过此时间未提交则回滚重分配）
	pendingTaskTimeout = 5 * time.Minute
)

// Match 匹配结果
type Match struct {
	JobID             string    `json:"job_id"`
	Mnemonic          string    `json:"mnemonic,omitempty"`           // 明文（未启用公钥时）
	EncryptedMnemonic string    `json:"encrypted_mnemonic,omitempty"` // RSA-OAEP base64（启用公钥后），服务端不可解密
	Address           string    `json:"address"`                      // TRON Base58Check 地址
	RawAddrHex        string    `json:"raw_addr_hex"`                 // 20字节地址的 hex 编码，用于重启后恢复查询能力
	Time              time.Time `json:"time"`
	Exists            bool      `json:"exists"` // 账户是否存在于数据库中
	rawAddr           []byte    // 运行时原始20字节地址（从 RawAddrHex 恢复）
}

// MatchView API 响应用，不暴露 RawAddrHex 等内部字段
type MatchView struct {
	JobID             string    `json:"job_id"`
	Mnemonic          string    `json:"mnemonic,omitempty"`
	EncryptedMnemonic string    `json:"encrypted_mnemonic,omitempty"`
	Address           string    `json:"address"`
	Time              time.Time `json:"time"`
	Exists            bool      `json:"exists"`
}

func matchToView(m *Match) MatchView {
	return MatchView{
		JobID:             m.JobID,
		Mnemonic:          m.Mnemonic,
		EncryptedMnemonic: m.EncryptedMnemonic,
		Address:           m.Address,
		Time:              m.Time,
		Exists:            m.Exists,
	}
}

// loadRSAPublicKey 从 PEM 文件加载 RSA 公钥（PKIX/SubjectPublicKeyInfo 格式）
func loadRSAPublicKey(path string) (*rsa.PublicKey, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("读取公钥文件失败: %w", err)
	}
	return parseRSAPublicKey(data)
}

// parseRSAPublicKeyPEM 从 PEM 字符串解析 RSA 公钥
func parseRSAPublicKeyPEM(pemStr string) (*rsa.PublicKey, error) {
	return parseRSAPublicKey([]byte(pemStr))
}

func parseRSAPublicKey(data []byte) (*rsa.PublicKey, error) {
	block, _ := pem.Decode(data)
	if block == nil {
		return nil, fmt.Errorf("无法解析 PEM 块")
	}
	pub, err := x509.ParsePKIXPublicKey(block.Bytes)
	if err != nil {
		return nil, fmt.Errorf("解析公钥失败: %w", err)
	}
	rsaPub, ok := pub.(*rsa.PublicKey)
	if !ok {
		return nil, fmt.Errorf("公钥不是 RSA 类型")
	}
	return rsaPub, nil
}

// encryptMnemonicRSA 使用 RSA-OAEP-SHA256 加密助记词，返回 base64 密文
// 加密后服务端不保留明文；只有持有对应私钥的人可解密。
func encryptMnemonicRSA(pub *rsa.PublicKey, plaintext string) (string, error) {
	ct, err := rsa.EncryptOAEP(sha256.New(), rand.Reader, pub, []byte(plaintext), nil)
	if err != nil {
		return "", err
	}
	return base64.StdEncoding.EncodeToString(ct), nil
}

func main() {
	flag.Parse()

	// 加载内嵌 RSA 公钥（编译时固定，服务端不持有私钥）
	pubKey, err := parseRSAPublicKeyPEM(embeddedPublicKey)
	if err != nil {
		log.Fatalf("内嵌公钥解析失败: %v", err)
	}
	log.Printf("🔒 已启用助记词加密存储（RSA-4096 OAEP），服务端不持有私钥")

	// 创建任务管理器
	tm, err := scheduler.NewTaskManager(*dataDir)
	if err != nil {
		log.Fatalf("创建任务管理器失败: %v", err)
	}

	// 创建服务器
	server := &Server{
		taskManager:   tm,
		matchesFile:   filepath.Join(*dataDir, "matches.json"),
		confirmedFile: filepath.Join(*dataDir, "confirmed.json"),
		pubKey:        pubKey,
		sessions:      make(map[string]time.Time),
		jobs:          make(map[string]*CompactJobRunner),
		pendingTasks:  make(map[int64]pendingTaskRecord),
		workers:       make(map[string]*WorkerInfo),
		confirmed:     make([]*Match, 0),
		seedComputer:  compute.NewCPUComputer(), // 用于验证提交的地址
	}

	// 解析 -auth 参数
	if *authFlag != "" {
		parts := strings.SplitN(*authFlag, ":", 2)
		if len(parts) != 2 || parts[0] == "" || parts[1] == "" {
			log.Fatalf("-auth 格式错误，应为 username:password")
		}
		server.authUser = parts[0]
		server.authPass = parts[1]
		log.Printf("🔐 Web UI 认证已启用，用户: %s", server.authUser)
	}

	// 初始化账户数据库
	if *accountDb != "" {
		adb, err := account.NewAccountDb(*accountDb)
		if err != nil {
			log.Fatalf("账户数据库初始化失败: %v", err)
		}
		server.accountDb = adb
	} else {
		panic("未指定 -accountdb 参数，匹配结果将无法确认账户是否存在")
	}

	// 恢复任务
	server.restoreJobs()

	// 加载持久化的确认成功地址（先加载）
	server.loadConfirmed()

	// 加载持久化的匹配结果（会将已确认的同步到 confirmed 列表）
	server.loadMatches()

	// Worker清理
	go server.cleanWorkers()

	// 路由
	http.HandleFunc("/login", server.handleLoginPage)
	http.HandleFunc("/api/login", server.handleLogin)
	http.HandleFunc("/api/logout", server.handleLogout)
	// Worker 路由不需要认证
	http.HandleFunc("/api/task/fetch", server.handleTaskFetch)
	http.HandleFunc("/api/task/submit", server.handleTaskSubmit)
	http.HandleFunc("/api/template", server.handleTemplate)
	// Web UI / Admin 路由需要认证
	http.HandleFunc("/", server.withAuth(server.handleIndex))
	http.HandleFunc("/api/jobs", server.withAuth(server.handleJobs))
	http.HandleFunc("/api/jobs/", server.withAuth(server.handleJobAction))
	http.HandleFunc("/api/matches", server.withAuth(server.handleMatches))
	http.HandleFunc("/api/confirmed", server.withAuth(server.handleConfirmed))
	http.HandleFunc("/api/workers", server.withAuth(server.handleWorkers))
	http.HandleFunc("/api/stats", server.withAuth(server.handleStats))
	http.HandleFunc("/api/bloom", server.withAuth(server.handleBloomInfo))

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

// withAuth 认证中间件：未启用认证时直接透传；启用时校验 session cookie
func (s *Server) withAuth(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if s.authUser == "" {
			next(w, r)
			return
		}
		if s.checkSession(r) {
			next(w, r)
			return
		}
		// API 请求返回 401，页面请求重定向到登录页
		if strings.HasPrefix(r.URL.Path, "/api/") {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusUnauthorized)
			w.Write([]byte(`{"error":"未登录"}`))
			return
		}
		http.Redirect(w, r, "/login", http.StatusFound)
	}
}

// checkSession 验证 session cookie 是否有效（24小时有效期）
func (s *Server) checkSession(r *http.Request) bool {
	cookie, err := r.Cookie("boon_session")
	if err != nil {
		return false
	}
	s.sessionsMu.Lock()
	defer s.sessionsMu.Unlock()
	t, ok := s.sessions[cookie.Value]
	if !ok {
		return false
	}
	if time.Since(t) > 24*time.Hour {
		delete(s.sessions, cookie.Value)
		return false
	}
	return true
}

// handleLogin POST /api/login — 验证凭据并设置 session cookie
func (s *Server) handleLogin(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "Method not allowed", 405)
		return
	}
	var req struct {
		Username string `json:"username"`
		Password string `json:"password"`
	}
	json.NewDecoder(r.Body).Decode(&req)

	w.Header().Set("Content-Type", "application/json")

	if s.authUser == "" || (req.Username == s.authUser && req.Password == s.authPass) {
		token := make([]byte, 24)
		rand.Read(token)
		sid := hex.EncodeToString(token)
		s.sessionsMu.Lock()
		s.sessions[sid] = time.Now()
		s.sessionsMu.Unlock()
		http.SetCookie(w, &http.Cookie{
			Name:     "boon_session",
			Value:    sid,
			Path:     "/",
			MaxAge:   86400,
			HttpOnly: true,
			SameSite: http.SameSiteLaxMode,
		})
		w.Write([]byte(`{"ok":true}`))
		return
	}
	w.WriteHeader(http.StatusUnauthorized)
	w.Write([]byte(`{"error":"用户名或密码错误"}`))
}

// handleLogout POST /api/logout — 注销 session
func (s *Server) handleLogout(w http.ResponseWriter, r *http.Request) {
	cookie, err := r.Cookie("boon_session")
	if err == nil {
		s.sessionsMu.Lock()
		delete(s.sessions, cookie.Value)
		s.sessionsMu.Unlock()
	}
	http.SetCookie(w, &http.Cookie{
		Name:   "boon_session",
		Value:  "",
		Path:   "/",
		MaxAge: -1,
	})
	http.Redirect(w, r, "/login", http.StatusFound)
}

// handleLoginPage GET /login — 返回登录页面
func (s *Server) handleLoginPage(w http.ResponseWriter, r *http.Request) {
	data, _ := staticFS.ReadFile("static/login.html")
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
			completedIdx int64
			totalIdx     int64
			running      bool
			speedWindow  []jobSpeedEntry
			gapCount     int
			gapIndices   int64
		}
		snaps := make(map[string]runnerSnap, len(s.jobs))
		for id, r := range s.jobs {
			win := make([]jobSpeedEntry, len(r.speedWindow))
			copy(win, r.speedWindow)
			var gapIndices int64
			for _, g := range r.gaps {
				gapIndices += g.end - g.start
			}
			snaps[id] = runnerSnap{
				completedIdx: r.CompletedIdx,
				totalIdx:     r.TotalIdx,
				running:      r.Running,
				speedWindow:  win,
				gapCount:     len(r.gaps),
				gapIndices:   gapIndices,
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

				// 速度 & ETA：用墙钟滑动窗口计算（totalIdx / 真实流逝时间）
				// 旧做法用 sum(worker耗时) 作分母，多 Worker 并行时会被倍数低估
				if snap.running && len(snap.speedWindow) > 0 {
					cutoff := now.Add(-60 * time.Second)
					var totalIdx int64
					var windowStart time.Time
					for _, e := range snap.speedWindow {
						if !e.t.Before(cutoff) {
							totalIdx += e.indices
							if windowStart.IsZero() || e.t.Before(windowStart) {
								windowStart = e.t
							}
						}
					}
					if totalIdx > 0 && !windowStart.IsZero() {
						windowSecs := now.Sub(windowStart).Seconds()
						if windowSecs > 0 {
							v.Speed = int64(float64(totalIdx) / windowSecs)
							if v.Speed > 0 {
								remaining := snap.totalIdx - snap.completedIdx
								v.ETASeconds = remaining / v.Speed
							}
						}
					}
				}

				// 缺口信息
				v.GapCount = int64(snap.gapCount)
				v.GapIndices = snap.gapIndices
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

	// 更新Worker（记录速度）；检测新 worker 上线或离线 worker 重连
	s.workersMu.Lock()
	wasNew := false
	if info, exists := s.workers[workerID]; exists {
		// 若上次活跃超过60s，视为重连
		wasNew = time.Since(info.LastSeen) > 60*time.Second
		info.LastSeen = time.Now()
		info.Speed = speed
	} else {
		s.workers[workerID] = &WorkerInfo{
			ID:       workerID,
			LastSeen: time.Now(),
			Speed:    speed,
		}
		wasNew = true
	}
	newCount := len(s.workers)
	s.workersMu.Unlock()

	// 新 worker 加入：更新计数并重置速度窗口
	if wasNew {
		s.jobsMu.Lock()
		if newCount != s.activeWorkerCount {
			log.Printf("[Scheduler] 新 Worker 上线: %s，在线数: %d → %d，重置速度窗口", workerID, s.activeWorkerCount, newCount)
			s.activeWorkerCount = newCount
			s.resetJobSpeedWindowsLocked()
		}
		s.jobsMu.Unlock()
	}

	// 分配任务
	s.jobsMu.Lock()
	defer s.jobsMu.Unlock()

	for jobID, runner := range s.jobs {
		if !runner.Running {
			continue
		}

		var start, end int64

		// 优先从缺口队列分配
		if len(runner.gaps) > 0 {
			gap := runner.gaps[0]
			start = gap.start
			end = gap.end
			if end-start > batchSize {
				// 缺口太大，只分配一部分
				end = start + batchSize
				runner.gaps[0].start = end
			} else {
				// 缺口全部分配，移除
				runner.gaps = runner.gaps[1:]
			}
		} else {
			// 无缺口，从 CurrentIdx 分配
			start = runner.CurrentIdx
			if start >= runner.TotalIdx {
				continue
			}

			end = start + batchSize
			if end > runner.TotalIdx {
				end = runner.TotalIdx
			}
			runner.CurrentIdx = end
		}

		task := &protocol.CompactTask{
			TaskID:   time.Now().UnixNano(),
			JobID:    runner.Template.JobID,
			StartIdx: start,
			EndIdx:   end,
		}

		// 记录待确认任务（内存 + 持久化）
		now := time.Now()
		s.pendingTasksMu.Lock()
		s.pendingTasks[task.TaskID] = pendingTaskRecord{jobID: jobID, startIdx: start, endIdx: end, fetchedAt: now}
		s.pendingTasksMu.Unlock()

		// 持久化 pending task
		s.taskManager.AddPendingTask(scheduler.PendingTask{
			JobID:     jobID,
			StartIdx:  start,
			EndIdx:    end,
			TaskID:    task.TaskID,
			FetchedAt: now.Unix(),
		})

		w.Header().Set("Content-Type", "application/octet-stream")
		w.Write(task.Encode())

		if len(runner.gaps) > 0 || start < runner.CurrentIdx {
			log.Printf("[Task] %s <- job=%s range=[%d,%d) (缺口填充) gap剩余=%d",
				workerID, jobID, start, end, len(runner.gaps))
		} else {
			log.Printf("[Task] %s <- job=%s range=[%d,%d) batch=%d speed=%d/s",
				workerID, jobID, start, end, end-start, speed)
		}
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

	// 检查是否有错误数据（index == 0 是无效的）
	for _, item := range result.Matches {
		if item.Index == 0 {
			log.Printf("[Submit] Worker %s 提交了无效数据 (index=0)，拒绝并保留缺口", workerID)
			w.WriteHeader(http.StatusOK)
			return // 不删除 pending task，等待超时后自动加入缺口
		}
	}

	// 更新Worker
	s.workersMu.Lock()
	if w, exists := s.workers[workerID]; exists {
		w.TasksDone++
		w.LastSeen = time.Now()
	}
	s.workersMu.Unlock()

	// 获取 pending task 记录（但不立即删除）
	s.pendingTasksMu.Lock()
	rec, found := s.pendingTasks[result.TaskID]
	s.pendingTasksMu.Unlock()

	if !found {
		// 任务不存在（可能已超时被加入缺口），直接返回
		w.WriteHeader(http.StatusOK)
		return
	}

	// ============================================================
	// 验证阶段：先验证所有匹配，再决定是否标记任务完成
	// ============================================================
	type verifiedMatch struct {
		data     protocol.MatchData
		mnemonic string
	}
	validMatches := make([]verifiedMatch, 0, len(result.Matches))
	verificationFailed := false

	for _, match := range result.Matches {
		// 1. 验证索引范围
		if match.Index < rec.startIdx || match.Index >= rec.endIdx {
			log.Printf("[Verify] ❌ Worker %s 提交的索引超出范围! index=%d, range=[%d,%d)",
				workerID, match.Index, rec.startIdx, rec.endIdx)
			verificationFailed = true
			continue
		}

		// 在 job 还未删除时还原助记词，并携带到后续存储阶段
		mnemonicStr := s.indexToMnemonic(rec.jobID, match.Index)

		// 2. 验证地址正确性
		if !s.verifyMatch(mnemonicStr, match.Address) {
			log.Printf("[Verify] ❌ Worker %s 提交的数据验证失败! index=%d, submittedAddr=%x",
				workerID, match.Index, match.Address)
			verificationFailed = true
			continue
		}

		validMatches = append(validMatches, verifiedMatch{data: match, mnemonic: mnemonicStr})
	}

	// 如果有验证失败，不删除 pending task，让它自然超时后加入缺口队列
	// pending task 已持久化，超时机制会将其加入 gaps（也会持久化恢复）
	if verificationFailed {
		log.Printf("[Verify] ⚠️ Worker %s 提交验证失败，索引 [%d, %d) 将等待超时后加入缺口队列",
			workerID, rec.startIdx, rec.endIdx)
		// 不删除 pending task，保留让它超时后自动处理
		w.WriteHeader(http.StatusOK)
		return
	}

	// ============================================================
	// 验证通过，标记任务完成
	// ============================================================
	s.pendingTasksMu.Lock()
	delete(s.pendingTasks, result.TaskID)
	s.taskManager.CompletePendingTask(result.TaskID)
	// 计算此 job 的连续完成前沿
	committedByJob := make(map[string]int64)
	for taskID, r := range s.pendingTasks {
		if taskID == result.TaskID {
			continue
		}
		if cur, ok := committedByJob[r.jobID]; !ok || r.startIdx < cur {
			committedByJob[r.jobID] = r.startIdx
		}
	}
	s.pendingTasksMu.Unlock()

	s.jobsMu.Lock()
	for jobID, runner := range s.jobs {
		if jobID == rec.jobID {
			indices := rec.endIdx - rec.startIdx
			runner.CompletedIdx += indices
			// 记录到1分钟滑动窗口
			now := time.Now()
			taskDur := now.Sub(rec.fetchedAt)
			if taskDur <= 0 {
				taskDur = time.Millisecond
			}
			cutoff := now.Add(-60 * time.Second)
			i := 0
			for i < len(runner.speedWindow) && runner.speedWindow[i].t.Before(cutoff) {
				i++
			}
			runner.speedWindow = append(runner.speedWindow[i:], jobSpeedEntry{t: now, indices: indices, duration: taskDur})
			// 更新安全重启点
			if minPending, ok := committedByJob[jobID]; ok {
				runner.CommittedIdx = minPending
			} else {
				runner.CommittedIdx = runner.CurrentIdx
			}
			s.taskManager.SetCompleted(jobID, runner.CurrentIdx)

			// 清理重叠的缺口
			runner.gaps = removeOverlappingGaps(runner.gaps, rec.startIdx, rec.endIdx)

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

	// ============================================================
	// 处理有效匹配
	// ============================================================
	for _, vm := range validMatches {
		mnemonicStr := vm.mnemonic
		tronAddr := bip44.GetTronAddress(vm.data.Address)

		m := &Match{
			Address:    tronAddr,
			RawAddrHex: hex.EncodeToString(vm.data.Address),
			Time:       time.Now(),
			rawAddr:    vm.data.Address,
		}

		if s.pubKey != nil {
			// 启用非对称加密：明文加密后立即丢弃，服务端不保留
			enc, err := encryptMnemonicRSA(s.pubKey, mnemonicStr)
			if err != nil {
				log.Printf("[Match] 加密助记词失败: %v", err)
				continue
			}
			m.EncryptedMnemonic = enc
		} else {
			m.Mnemonic = mnemonicStr
		}

		s.matchesMu.Lock()
		s.matches = append(s.matches, m)
		s.saveMatchesLocked()
		s.matchesMu.Unlock()

		log.Printf("========== 匹配 ✅已验证 ==========")
		log.Printf("Worker: %s", workerID)
		if s.pubKey != nil {
			log.Printf("助记词: [已加密，仅私钥持有人可解密]")
		} else {
			log.Printf("助记词: %s", mnemonicStr)
		}
		log.Printf("地址: %s", tronAddr)
		log.Printf("====================================")

		// 后台异步确认账户状态
		go s.confirmMatch(m)
	}

	w.WriteHeader(http.StatusOK)
}

// indexToMnemonic 从索引还原助记词
func (s *Server) indexToMnemonic(jobID string, idx int64) string {
	s.jobsMu.RLock()
	defer s.jobsMu.RUnlock()

	runner, ok := s.jobs[jobID]
	if !ok {
		return ""
	}

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

// handleMatches 匹配结果
func (s *Server) handleMatches(w http.ResponseWriter, r *http.Request) {
	//s.matchesMu.Lock()
	//views := make([]MatchView, len(s.matches))
	//for i, m := range s.matches {
	//	views[i] = matchToView(m)
	//}
	//s.matchesMu.Unlock()
	//w.Header().Set("Content-Type", "application/json")
	//json.NewEncoder(w).Encode(map[string]interface{}{"matches": views})
}

// handleConfirmed 确认成功的地址
func (s *Server) handleConfirmed(w http.ResponseWriter, r *http.Request) {
	s.confirmedMu.Lock()
	views := make([]MatchView, len(s.confirmed))
	for i, m := range s.confirmed {
		views[i] = matchToView(m)
	}
	s.confirmedMu.Unlock()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{"confirmed": views})
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
		"encrypted": s.pubKey != nil, // 是否启用了非对称加密
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

		// 从持久化加载该 job 的 pending tasks 作为缺口
		pendingTasks := s.taskManager.GetPendingTasksByJob(job.ID)
		gaps := make([]indexRange, 0, len(pendingTasks))
		for _, pt := range pendingTasks {
			gaps = append(gaps, indexRange{start: pt.StartIdx, end: pt.EndIdx})
		}

		runner := &CompactJobRunner{
			Job:               job,
			Template:          template,
			CurrentIdx:        job.Completed, // 从已分发位置继续
			CompletedIdx:      0,
			CommittedIdx:      job.Completed,
			TotalIdx:          total,
			BatchSize:         int64(job.BatchSize),
			StopCh:            make(chan struct{}),
			Running:           false, // 暂停状态，等待用户恢复
			ActiveSince:       time.Time{},
			CompletedAtActive: 0,
			gaps:              gaps,
		}

		s.jobsMu.Lock()
		s.jobs[job.ID] = runner
		s.jobsMu.Unlock()

		log.Printf("任务已恢复（暂停）: %s 进度=%d/%d", job.ID, job.Completed, total)
	}
}

// cleanWorkers 清理Worker，处理超时任务，并在在线数量变化时重置 job 速度窗口
func (s *Server) cleanWorkers() {
	ticker := time.NewTicker(10 * time.Second)
	for range ticker.C {
		now := time.Now()
		workerCutoff := now.Add(-60 * time.Second)

		// 清理超时 Worker
		s.workersMu.Lock()
		for id, w := range s.workers {
			if w.LastSeen.Before(workerCutoff) {
				delete(s.workers, id)
			}
		}
		newCount := len(s.workers)
		s.workersMu.Unlock()

		// 处理超时的飞行中任务（加入缺口队列）
		s.addTimedOutTasksToGaps(now)

		// 更新 worker 计数并重置速度窗口
		s.jobsMu.Lock()
		if newCount != s.activeWorkerCount {
			log.Printf("[Scheduler] 在线 Worker 数变化: %d → %d，重置速度窗口", s.activeWorkerCount, newCount)
			s.activeWorkerCount = newCount
			s.resetJobSpeedWindowsLocked()
		}
		s.jobsMu.Unlock()
	}
}

// addTimedOutTasksToGaps 将超时的飞行中任务加入缺口队列，优先重新分配
func (s *Server) addTimedOutTasksToGaps(now time.Time) {
	taskCutoff := now.Add(-pendingTaskTimeout)

	s.pendingTasksMu.Lock()
	defer s.pendingTasksMu.Unlock()

	if len(s.pendingTasks) == 0 {
		return
	}

	// 收集超时任务，按 jobID 分组
	timedOut := make(map[string][]pendingTaskRecord) // jobID -> []task
	var timedOutTaskIDs []int64
	for taskID, rec := range s.pendingTasks {
		if rec.fetchedAt.Before(taskCutoff) {
			timedOut[rec.jobID] = append(timedOut[rec.jobID], rec)
			timedOutTaskIDs = append(timedOutTaskIDs, taskID)
			delete(s.pendingTasks, taskID)
		}
	}

	if len(timedOut) == 0 {
		return
	}

	// 删除持久化的超时任务
	for _, taskID := range timedOutTaskIDs {
		s.taskManager.CompletePendingTask(taskID)
	}

	// 将超时任务加入对应 job 的缺口队列
	s.jobsMu.Lock()
	defer s.jobsMu.Unlock()

	for jobID, tasks := range timedOut {
		runner, ok := s.jobs[jobID]
		if !ok || !runner.Running {
			continue
		}

		var totalIndices int64
		for _, t := range tasks {
			runner.gaps = append(runner.gaps, indexRange{start: t.startIdx, end: t.endIdx})
			totalIndices += t.endIdx - t.startIdx
		}

		// 按 start 排序缺口，便于顺序分配
		sortGaps(runner.gaps)

		log.Printf("[Scheduler] 任务超时加入缺口: job=%s 缺口数=%d 索引数=%d",
			jobID, len(runner.gaps), totalIndices)
	}
}

// sortGaps 按起始位置排序缺口
func sortGaps(gaps []indexRange) {
	for i := 0; i < len(gaps)-1; i++ {
		for j := i + 1; j < len(gaps); j++ {
			if gaps[j].start < gaps[i].start {
				gaps[i], gaps[j] = gaps[j], gaps[i]
			}
		}
	}
}

// removeOverlappingGaps 移除与指定区间重叠的缺口
func removeOverlappingGaps(gaps []indexRange, start, end int64) []indexRange {
	var result []indexRange
	for _, g := range gaps {
		if g.end <= start || g.start >= end {
			// 无重叠
			result = append(result, g)
		} else {
			// 有重叠，可能需要分割
			if g.start < start {
				result = append(result, indexRange{start: g.start, end: start})
			}
			if g.end > end {
				result = append(result, indexRange{start: end, end: g.end})
			}
		}
	}
	return result
}

// resetJobSpeedWindowsLocked 清空所有 job 的速度窗口（调用前须持有 jobsMu 写锁）
func (s *Server) resetJobSpeedWindowsLocked() {
	for _, runner := range s.jobs {
		runner.speedWindow = runner.speedWindow[:0]
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

	// 从 RawAddrHex 恢复 rawAddr
	for _, m := range matches {
		if m.RawAddrHex != "" {
			if b, err := hex.DecodeString(m.RawAddrHex); err == nil {
				m.rawAddr = b
			}
		}
	}

	s.matchesMu.Lock()
	s.matches = matches
	// 将已确认的匹配添加到 confirmed 列表（在 confirmedMu 中）
	var confirmedFromMatches []*Match
	for _, m := range matches {
		if m.Exists {
			confirmedFromMatches = append(confirmedFromMatches, m)
		}
	}
	s.matchesMu.Unlock()

	// 将已确认的匹配同步到 confirmed 列表
	if len(confirmedFromMatches) > 0 {
		s.confirmedMu.Lock()
		// 建立地址集合，避免重复
		existingAddrs := make(map[string]bool)
		for _, c := range s.confirmed {
			existingAddrs[c.Address] = true
		}
		for _, m := range confirmedFromMatches {
			if !existingAddrs[m.Address] {
				s.confirmed = append(s.confirmed, m)
				existingAddrs[m.Address] = true
			}
		}
		s.saveConfirmedLocked()
		s.confirmedMu.Unlock()
	}

	log.Printf("[Matches] 已加载 %d 条匹配记录", len(matches))
}

// loadConfirmed 从磁盘恢复已确认成功的地址
func (s *Server) loadConfirmed() {
	if s.confirmedFile == "" {
		return
	}
	data, err := os.ReadFile(s.confirmedFile)
	if err != nil {
		if !os.IsNotExist(err) {
			log.Printf("[Confirmed] 读取失败: %v", err)
		}
		return
	}

	var confirmed []*Match
	if err := json.Unmarshal(data, &confirmed); err != nil {
		log.Printf("[Confirmed] 解析失败: %v", err)
		return
	}

	// 从 RawAddrHex 恢复 rawAddr
	for _, m := range confirmed {
		if m.RawAddrHex != "" {
			if b, err := hex.DecodeString(m.RawAddrHex); err == nil {
				m.rawAddr = b
			}
		}
	}

	s.confirmedMu.Lock()
	s.confirmed = confirmed
	s.confirmedMu.Unlock()

	log.Printf("[Confirmed] 已加载 %d 条确认成功记录", len(confirmed))
}

// saveConfirmedLocked 将确认成功的地址写入磁盘（调用前须持有 confirmedMu）
func (s *Server) saveConfirmedLocked() {
	if s.confirmedFile == "" {
		return
	}
	data, err := json.MarshalIndent(s.confirmed, "", "  ")
	if err != nil {
		log.Printf("[Confirmed] 序列化失败: %v", err)
		return
	}
	if err := os.WriteFile(s.confirmedFile, data, 0644); err != nil {
		log.Printf("[Confirmed] 写入失败: %v", err)
	}
}

// extractJobNum 提取job数字
func extractJobNum(jobID string) int64 {
	var num int64
	fmt.Sscanf(jobID, "job-%d", &num)
	return num
}

// verifyMatch 验证提交的地址是否正确（重新计算地址并比较）
func (s *Server) verifyMatch(mnemonicStr string, submittedAddr []byte) bool {
	s.verificationMu.Lock()
	defer s.verificationMu.Unlock()

	// 使用 CPU 计算器重新计算地址
	computedAddrs := s.seedComputer.Compute([]string{mnemonicStr})
	if len(computedAddrs) == 0 || len(computedAddrs[0]) != 20 {
		log.Printf("[Verify] 计算地址失败: mnemonic=%s", mnemonicStr)
		return false
	}

	// 比较计算结果与提交的地址
	computedAddr := computedAddrs[0]
	if !bytes.Equal(computedAddr, submittedAddr) {
		log.Printf("[Verify] 地址不匹配! computed=%x, submitted=%x", computedAddr, submittedAddr)
		return false
	}

	return true
}

// confirmMatch 检查账户是否存在于本地数据库中
func (s *Server) confirmMatch(m *Match) {
	if s.accountDb == nil {
		return
	}

	exists := s.accountDb.IsExist(m.rawAddr)

	s.matchesMu.Lock()
	m.Exists = exists
	s.saveMatchesLocked()
	s.matchesMu.Unlock()

	if exists {
		// 添加到确认成功列表
		s.confirmedMu.Lock()
		// 检查是否已存在（避免重复）
		found := false
		for _, c := range s.confirmed {
			if c.Address == m.Address {
				found = true
				break
			}
		}
		if !found {
			s.confirmed = append(s.confirmed, m)
			s.saveConfirmedLocked()
		}
		s.confirmedMu.Unlock()
		log.Printf("[Confirm] ✅ 账户存在 %s", m.Address)
	} else {
		log.Printf("[Confirm] ❌ 账户不存在 %s", m.Address)
	}
}
