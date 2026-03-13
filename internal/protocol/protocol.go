package protocol

// Task 任务
type Task struct {
	ID        int      `json:"id"`
	Mnemonics []string `json:"mnemonics"`
}

// Result 结果
type Result struct {
	TaskID    int      `json:"task_id"`
	Addresses [][]byte `json:"addresses"` // 20 bytes each
	Mnemonics []string `json:"mnemonics"` // 原始助记词（用于匹配结果）
}

// TaskRequest 任务请求
type TaskRequest struct {
	WorkerID string `json:"worker_id"`
}

// TaskResponse 任务响应
type TaskResponse struct {
	Task  *Task `json:"task"`
	Count int   `json:"count"` // 任务数量，0表示没有更多任务
}

// ResultRequest 结果提交请求
type ResultRequest struct {
	WorkerID  string   `json:"worker_id"`
	TaskID    int      `json:"task_id"`
	Addresses [][]byte `json:"addresses"`
}

// ResultResponse 结果提交响应
type ResultResponse struct {
	Success bool `json:"success"`
	Matches int  `json:"matches"` // 匹配数量
}

// Match 匹配结果
type Match struct {
	Mnemonic string `json:"mnemonic"`
	Address  []byte `json:"address"`
}
