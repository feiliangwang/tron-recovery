# Boon

TRON 助记词分布式枚举与地址恢复工具。

## 架构

```
┌─────────────────┐       ┌─────────────────┐
│   Scheduler     │       │     Worker      │
│   (调度端)       │◄─────►│   (计算端)       │
├─────────────────┤ HTTP  ├─────────────────┤
│ • 助记词枚举     │       │ • 获取任务       │
│ • BIP39验证     │       │ • PBKDF2计算     │
│ • 任务分发      │       │ • BIP44派生      │
│ • 结果接收      │       │ • Keccak256      │
│ • Bloom过滤     │       │ • 提交结果       │
└─────────────────┘       └─────────────────┘
        ▲                         ▲
        │                         │
        │                  可以启动多个
        │                  (支持GPU加速)
```

**两个独立组件：**

1. **Scheduler（调度端）**
   - 枚举助记词（`?` 占位符）
   - 验证 BIP39 合法性
   - 分发任务给 Worker
   - 接收结果并 Bloom 过滤
   - 记录匹配结果

2. **Worker（计算端）**
   - 从 Scheduler 获取任务
   - 计算 Hash 地址
   - 提交结果给 Scheduler
   - 支持分布式部署
   - 可选 GPU 加速

## 编译

```bash
# 构建所有组件
make build

# 或单独构建
make scheduler  # 调度器
make worker     # Worker（CPU）
make worker-gpu # Worker（GPU，需要CUDA）
```

## 使用

### 1. 启动调度器

```bash
./build/scheduler \
  -mnemonic "word1 word2 ? ? word5 word6 word7 word8 word9 word10 word11 word12" \
  -bloom addresses.txt \
  -port 8080
```

**参数：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-mnemonic` | 助记词模板，`?` 表示未知词 | 必填 |
| `-bloom` | Bloom 过滤器文件 | 可选 |
| `-batch` | 批次大小 | 1000 |
| `-port` | HTTP 服务端口 | 8080 |
| `-o` | 匹配结果输出文件 | matches.txt |

### 2. 启动 Worker

```bash
# 单机启动
./build/worker -scheduler http://localhost:8080

# 多机分布式
# 机器A
./build/worker -scheduler http://scheduler-ip:8080 -id worker-A

# 机器B（GPU加速）
./build/worker-gpu -scheduler http://scheduler-ip:8080 -id worker-B
```

**参数：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-scheduler` | 调度服务器地址 | `http://localhost:8080` |
| `-id` | Worker ID | 自动生成 |
| `-workers` | 并发数 | CPU核心数 |
| `-poll` | 轮询间隔 | 100ms |

### 3. 查看统计

```bash
curl http://localhost:8080/api/stats
```

输出：
```json
{
  "total_tasks": 1000,
  "completed_tasks": 800,
  "pending_tasks": 200,
  "matches": 5,
  "elapsed": "2m30s",
  "rate": 5.33
}
```

## API 接口

### 获取任务

```
POST /api/task/fetch
Content-Type: application/json

{"worker_id": "worker-1"}
```

响应：
```json
{
  "task": {
    "id": 123,
    "mnemonics": [["word1", "word2", ...], ...]
  },
  "count": 99
}
```

### 提交结果

```
POST /api/task/submit
Content-Type: application/json

{
  "worker_id": "worker-1",
  "task_id": 123,
  "addresses": ["a1b2c3d4...", "e5f6a7b8...", ...]
}
```

### 查看统计

```
GET /api/stats
```

## Bloom 过滤器文件格式

Bloom 过滤器使用 gob 编码格式存储。

### 生成 Bloom 文件

```bash
# 从地址列表生成 Bloom 过滤器
./build/bloomtool -input addresses.txt -output bloom.gob
```

**输入文件格式**（每行一个 hex 地址）：

```
a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2
b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3
...
```

**输出文件**: gob 编码的 Bloom 过滤器

### 在调度器中使用

```bash
./build/scheduler -bloom bloom.gob
```

或在 Web UI 创建任务时填写 Bloom 文件路径。

## 性能估算

| 未知词数 | 组合数 | 建议配置 |
|----------|--------|----------|
| 1 | 2,048 | 单机 CPU |
| 2 | 4,194,304 | 2-4 Worker |
| 3 | 8.5B | 10+ Worker + GPU |

## 许可证

MIT
