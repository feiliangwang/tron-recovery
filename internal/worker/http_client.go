package worker

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"boon/internal/protocol"
)

// HTTPClient HTTP调度器客户端
type HTTPClient struct {
	baseURL    string
	httpClient *http.Client
}

// NewHTTPClient 创建HTTP客户端
func NewHTTPClient(baseURL string) *HTTPClient {
	return &HTTPClient{
		baseURL:    baseURL,
		httpClient: &http.Client{},
	}
}

// FetchTask 获取任务
func (c *HTTPClient) FetchTask(workerID string) (*protocol.Task, error) {
	url := fmt.Sprintf("%s/api/task/fetch", c.baseURL)

	reqBody := protocol.TaskRequest{WorkerID: workerID}
	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequest("POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusNoContent {
		return nil, nil // 没有任务
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("HTTP error: %d", resp.StatusCode)
	}

	var taskResp protocol.TaskResponse
	if err := json.NewDecoder(resp.Body).Decode(&taskResp); err != nil {
		return nil, err
	}

	return taskResp.Task, nil
}

// SubmitResult 提交结果
func (c *HTTPClient) SubmitResult(workerID string, result *protocol.Result) error {
	url := fmt.Sprintf("%s/api/task/submit", c.baseURL)

	// 将 [][]byte 转换为 hex string 数组以便 JSON 序列化
	type submitRequest struct {
		WorkerID  string   `json:"worker_id"`
		TaskID    int      `json:"task_id"`
		Addresses []string `json:"addresses"`
		Mnemonics []string `json:"mnemonics"`
	}

	addresses := make([]string, len(result.Addresses))
	for i, addr := range result.Addresses {
		if addr != nil {
			addresses[i] = fmt.Sprintf("%x", addr)
		}
	}

	reqBody := submitRequest{
		WorkerID:  workerID,
		TaskID:    result.TaskID,
		Addresses: addresses,
		Mnemonics: result.Mnemonics,
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return err
	}

	req, err := http.NewRequest("POST", url, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("HTTP error: %d - %s", resp.StatusCode, respBody)
	}

	return nil
}
