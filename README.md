# Boon

TRON 助记词分布式枚举与地址恢复工具。针对已知部分助记词的场景，通过枚举 BIP39 词表组合暴力搜索目标地址。

## 架构

```
┌──────────────────────┐              ┌─────────────────────┐
│      Scheduler       │◄────HTTP────►│       Worker        │
│      (调度端)         │   紧凑二进制  │      (计算端)         │
├──────────────────────┤              ├─────────────────────┤
│ • Web UI（任务管理）  │              │ • 获取索引范围任务   │
│ • 任务分发/状态持久化  │              │ • 枚举索引→助记词    │
│ • Bloom 过滤验证      │              │ • PBKDF2/BIP44推导  │
│ • RSA-4096 加密存储   │              │ • Bloom 本地预过滤   │
│ • LevelDB 账户数据库  │              │ • CPU / GPU 支持    │
└──────────────────────┘              └─────────────────────┘
                                             可启动多个实例
                                           （支持多卡GPU加速）
```

## 密码学流程

```
index → 助记词（BIP39 词表模运算）
  → PBKDF2-HMAC-SHA512（2048 次迭代，空 passphrase）
  → BIP44 派生路径：m/44'/195'/0'/0/0（TRON）
  → Keccak256（未压缩公钥，去掉 0x04 前缀）
  → 取最后 20 字节 = TRON 地址
```

## 编译

```bash
make build        # 编译所有组件 → build/
make scheduler    # 仅编译调度器
make worker       # 仅编译 CPU Worker
make worker-gpu   # 编译 GPU Worker（需要 CUDA，sm_75）
make bloomtool    # 编译 Bloom 过滤器生成工具
make test         # 运行所有单元测试
make deps         # go mod download && go mod tidy
```

## 使用

### 1. 准备 Bloom 过滤器（可选）

从 LevelDB 账户数据库生成 Bloom 过滤器：

```bash
./build/bloomtool -input /path/to/accounts.leveldb -output bloom.gob -rate 0.001
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-input` | LevelDB 账户数据库路径 | 必填 |
| `-output` | 输出 gob 文件路径 | `bloom.gob` |
| `-rate` | 误判率 | `0.001` |

### 2. 启动调度器

```bash
./build/scheduler \
  -data ./data \
  -port 8080 \
  -accountdb /path/to/accounts.leveldb \
  -auth admin:password
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-data` | 状态数据目录 | `./data` |
| `-port` | HTTP 服务端口 | `8080` |
| `-accountdb` | LevelDB 账户数据库（用于二次验证） | 可选 |
| `-auth` | Web UI Basic Auth，格式 `user:pass` | 留空不启用 |

启动后访问 `http://localhost:8080` 通过 Web UI 创建任务（填写助记词模板，`?` 表示未知词）。

> 匹配到的助记词以 RSA-4096 公钥加密后存储，调度服务器不持有私钥，无法解密原文。
> 公钥常量 embeddedPublicKey,自行修改

### 3. 启动 Worker

**CPU 模式：**

```bash
./build/worker -scheduler http://192.168.3.250:8080 -id worker-1
```

**GPU 模式（单卡）：**

```bash
./build/worker-gpu -scheduler http://192.168.3.250:8080 -gpu
```

**GPU 模式（多卡）：**

```bash
./build/worker-gpu -scheduler http://192.168.3.250:8080 -gpu-devices 0,1,2
```

**使用本地 Bloom 过滤器预过滤（减少网络提交量）：**

```bash
./build/worker -scheduler http://192.168.3.250:8080 -bloom bloom.gob
```

**常用参数：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-scheduler` | 调度服务器地址 | `http://192.168.3.250` |
| `-id` | Worker ID | 主机名+PID |
| `-workers` | CPU 并发线程数 | CPU 核心数 |
| `-bloom` | 本地 Bloom 过滤器文件 | `account.bin.bloom` |
| `-gpu` | 启用所有可用 GPU | `false` |
| `-gpu-devices` | 指定 GPU 设备列表，如 `0,1,2` 或 `all` | — |

### 4. 查看统计

Web UI：`http://localhost:8080`

API：

```bash
curl http://localhost:8080/api/stats
```

## 测速与诊断工具

Worker 内置多种独立运行模式，无需连接调度器：

```bash
# 全链路测速（枚举→计算→验证，N 个索引）
./build/worker-gpu -bench-full 1048576 -gpu

# 纯 PBKDF2 GPU 吞吐测速
./build/worker-gpu -bench-pbkdf2 65536 -gpu

# GPU vs CPU 地址一致性验证
./build/worker-gpu -verify 10000

# Bloom 过滤器诊断（给定助记词，对比 CPU/GPU bloom 结果）
./build/worker-gpu -bloom-test "word1 word2 ... word12" -bloom bloom.gob -gpu

# BIP39 checksum GPU 调试
./build/worker-gpu -bip39-debug "word1 word2 ... word12" -gpu

# 枚举索引验证（验证 CPU/GPU 索引→助记词转换是否一致）
./build/worker-gpu -enum-test "word1 ... word12" -template "word1 ? ? word4 ..." -gpu
```

## 通信协议

调度器与 Worker 使用紧凑二进制协议（Protocol v2）：

- `CompactTask`：32 字节固定大小 `[TaskID:8][JobID:8][StartIdx:8][EndIdx:8]`
- 调度器每 Job 发送一次 `TaskTemplate`（含未知词位置和已知词）
- Worker 上报速度指标（addr/s），调度器据此自适应批次大小

## 性能参考

| 未知词数 | 搜索空间    | 建议配置             |
|------|---------|------------------|
| 1    | 2,048   | 单机 CPU，秒级完成      |
| 2    | ~420 万  | 2–4 Worker       |
| 3    | ~85 亿   | 10+ Worker + GPU |
| 4    | ~170 亿亿 | 多机多卡长期运行         |

## GPU性能参考

| 显卡     | 计算速度    | 3个未知词  | 4个未知词 |
|--------|---------|--------|-------|
| 3060   | 2.56M/s | ～1h    | ～85d  |
| 3060ti | 4.25M/s | ～0.5h  | ～43d  |
| 3070   | 3.6M/s  | ～0.66h | ～56d  |
| 3070ti | 4.35M/s | ～0.5h  | ～43d  |
| 4090   | 12.5M/s | ～0.2h  | ～17d  |

## 许可证

MIT
