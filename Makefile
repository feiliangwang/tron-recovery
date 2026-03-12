# Boon Makefile

# Go 参数
GOCMD=go
GOBUILD=$(GOCMD) build
GOCLEAN=$(GOCMD) clean
GOTEST=$(GOCMD) test
GOGET=$(GOCMD) get
GOMOD=$(GOCMD) mod

# CUDA 参数
NVCC=nvcc
CUDA_ARCH=sm_75
CUDA_INCLUDE=/usr/local/cuda/include
CUDA_LIB=/usr/local/cuda/lib64

# 项目结构
PROJECT_NAME=boon
BUILD_DIR=build

# 构建标志
LDFLAGS=-ldflags "-s -w"

.PHONY: all build clean test deps scheduler worker help

all: deps build

# 下载依赖
deps:
	$(GOMOD) download
	$(GOMOD) tidy

# 构建所有组件
build: scheduler worker bloomtool

# 构建调度器
scheduler:
	@mkdir -p $(BUILD_DIR)
	$(GOBUILD) $(LDFLAGS) -o $(BUILD_DIR)/scheduler ./cmd/scheduler

# 构建Worker（CPU版本）
worker:
	@mkdir -p $(BUILD_DIR)
	$(GOBUILD) $(LDFLAGS) -o $(BUILD_DIR)/worker ./cmd/worker

# 构建Bloom工具
bloomtool:
	@mkdir -p $(BUILD_DIR)
	$(GOBUILD) $(LDFLAGS) -o $(BUILD_DIR)/bloomtool ./cmd/bloomtool

# GPU版本Worker
worker-gpu: deps
	@echo "构建GPU版本Worker..."
	@mkdir -p $(BUILD_DIR)
	$(NVCC) -c -o $(BUILD_DIR)/compute.o internal/compute/gpu.cu -arch=$(CUDA_ARCH) -I$(CUDA_INCLUDE)
	CGO_LDFLAGS="-L$(CUDA_LIB) -lcuda -lcudart" \
	$(GOBUILD) $(LDFLAGS) -tags cuda -o $(BUILD_DIR)/worker-gpu ./cmd/worker

# 运行测试
test:
	$(GOTEST) -v ./...

# 清理
clean:
	$(GOCLEAN)
	rm -rf $(BUILD_DIR)

# 帮助
help:
	@echo "Boon - TRON助记词分布式枚举工具"
	@echo ""
	@echo "架构:"
	@echo "  Scheduler (调度器): 枚举任务 + 验证结果 + Bloom过滤"
	@echo "  Worker (计算端): 分布式计算Hash"
	@echo ""
	@echo "构建:"
	@echo "  make build        - 构建所有组件"
	@echo "  make scheduler    - 构建调度器"
	@echo "  make worker       - 构建Worker（CPU版本）"
	@echo "  make worker-gpu   - 构建Worker（GPU版本）"
	@echo "  make bloomtool    - 构建Bloom过滤器工具"
	@echo "  make test         - 运行测试"
	@echo "  make clean        - 清理构建文件"
	@echo ""
	@echo "使用:"
	@echo "  # 1. 生成Bloom过滤器文件"
	@echo "  ./build/bloomtool -input addresses.txt -output bloom.gob"
	@echo ""
	@echo "  # 2. 启动调度器"
	@echo "  ./build/scheduler -mnemonic 'word1 ? ? word4 ...' -bloom bloom.gob"
	@echo ""
	@echo "  # 3. 启动Worker（可以启动多个）"
	@echo "  ./build/worker -scheduler http://localhost:8080"
	@echo ""
	@echo "  # 4. 查看统计"
	@echo "  curl http://localhost:8080/api/stats"
	@echo ""
	@echo "Bloom文件格式:"
	@echo "  输入: 每行一个hex地址（20字节）"
	@echo "  输出: gob编码的Bloom过滤器"
