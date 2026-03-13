package tron

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"boon/internal/bip44"
)

const (
	DefaultEndpoint = "https://api.trongrid.io"
	httpTimeout     = 15 * time.Second
)

// Client TRON HTTP API 客户端
type Client struct {
	httpClient *http.Client
	baseURL    string
}

// NewClient 创建客户端（endpoint 为 HTTP base URL，如 "https://api.trongrid.io"）
func NewClient(endpoint string) (*Client, error) {
	return &Client{
		httpClient: &http.Client{Timeout: httpTimeout},
		baseURL:    endpoint,
	}, nil
}

// Close 关闭客户端（HTTP 无需显式关闭）
func (c *Client) Close() error { return nil }

// tronAccountResp TronGrid /v1/accounts/{address} 响应结构
type tronAccountResp struct {
	Data []struct {
		CreateTime int64 `json:"create_time"`
		Balance    int64 `json:"balance"`
	} `json:"data"`
}

// IsActivated 查询账户是否已激活（在链上存在）
// addr20: 20字节地址
// 返回: (activated bool, createTime int64, error)
// activated=true 表示账户在链上存在（data 非空）
func (c *Client) IsActivated(addr20 []byte) (bool, int64, error) {
	tronAddr := bip44.GetTronAddress(addr20)
	url := fmt.Sprintf("%s/v1/accounts/%s", c.baseURL, tronAddr)

	resp, err := c.httpClient.Get(url)
	if err != nil {
		return false, 0, fmt.Errorf("HTTP 请求失败: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return false, 0, fmt.Errorf("HTTP %d: %s", resp.StatusCode, body)
	}

	var result tronAccountResp
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return false, 0, fmt.Errorf("解析响应失败: %w", err)
	}

	// data 为空 = 账户不存在/未激活
	if len(result.Data) == 0 {
		return false, 0, nil
	}
	return true, result.Data[0].CreateTime, nil
}
