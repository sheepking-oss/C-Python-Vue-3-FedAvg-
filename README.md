# FedAvg 联邦学习本地沙盒系统

一个分布式机器学习模拟沙盒系统，包含三个核心组件：

- **C++ 聚合服务器**：高性能地接收多客户端权重张量并执行 FedAvg 参数聚合
- **Python GNN 客户端**：模拟边缘节点，加载隔离数据集进行图神经网络训练并上传权重
- **Vue 3 前端控制台**：通过 Web API 下发训练指令，实时绘制 Loss 收敛曲线

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                      Vue 3 前端控制台 (端口 3000)                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │ 训练控制面板 │  │ 客户端状态   │  │ Loss 收敛曲线实时绘制    │ │
│  │ (学习率配置) │  │ (实时监控)   │  │ (Chart.js)              │ │
│  └──────┬──────┘  └──────┬──────┘  └───────────┬─────────────┘ │
└─────────┼─────────────────┼──────────────────────┼───────────────┘
          │ HTTP/REST       │ HTTP/REST            │ HTTP/REST
          ▼                 ▼                      ▼
┌─────────────────────────────────────────────────────────────────┐
│              Python GNN 客户端 (端口 5001, 5002, 5003)          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │ client_1    │  │ client_2    │  │ client_3                │ │
│  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐             │ │
│  │ │ GCN模型 │ │  │ │ GCN模型 │ │  │ │ GCN模型 │             │ │
│  │ │ 隔离数据 │ │  │ │ 隔离数据 │ │  │ │ 隔离数据 │             │ │
│  │ └────┬────┘ │  │ └────┬────┘ │  │ └────┬────┘             │ │
│  └──────┼──────┘  └──────┼──────┘  └──────┼───────────────────┘ │
└─────────┼─────────────────┼─────────────────┼─────────────────────┘
          │ 上传权重         │ 上传权重         │ 上传权重
          │ 下载聚合权重      │ 下载聚合权重      │ 下载聚合权重
          ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                C++ 聚合服务器 (端口 8080)                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    FedAvg 聚合算法                        │    │
│  │  weight_avg = Σ(sample_count_i * weight_i) / Σ(samples) │    │
│  └─────────────────────────────────────────────────────────┘    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │ HTTP Server │  │ 张量存储    │  │ 训练轮次管理             │ │
│  │ (cpp-httplib)│  │ (std::vector)│  │ (round state)          │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## 快速开始

### 1. 环境准备

#### Python 环境
```bash
cd python_clients
pip install -r requirements.txt
```

注意：PyTorch Geometric 可能需要额外安装：
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch_geometric
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

#### C++ 环境
需要安装 CMake 3.14+ 和 C++17 编译器：
```bash
cd cpp_server
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

#### Node.js 环境
```bash
cd vue_frontend
npm install
```

### 2. 启动系统

按以下顺序启动各个组件：

#### 步骤 1: 启动 C++ 聚合服务器
```bash
cd cpp_server/build
./server.exe  # Windows
# 或 ./server  # Linux/Mac
```

服务器将在 `http://localhost:8080` 启动。

#### 步骤 2: 启动 Python GNN 客户端
```bash
cd python_clients
python start_clients.py
```

这将同时启动 3 个客户端，分别在以下端口：
- client_1: http://localhost:5001
- client_2: http://localhost:5002
- client_3: http://localhost:5003

#### 步骤 3: 启动 Vue 3 前端
```bash
cd vue_frontend
npm run dev
```

前端将在 `http://localhost:3000` 启动。

### 3. 开始训练

1. 打开浏览器访问 `http://localhost:3000`
2. 在「训练控制面板」中设置：
   - 学习率 (默认 0.01)
   - 训练轮数 (默认 10)
3. 点击「开始训练」按钮
4. 观察「Loss 收敛曲线」实时绘制
5. 所有客户端完成当前轮次后，可以点击「手动触发权重聚合」

## API 接口说明

### C++ 聚合服务器 (端口 8080)

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/submit` | POST | 客户端提交权重 |
| `/api/weights` | GET | 获取聚合后的全局权重 |
| `/api/status` | GET | 获取服务器状态 |
| `/api/aggregate` | POST | 手动触发聚合 |
| `/api/config` | POST | 配置服务器参数 |
| `/api/reset` | POST | 重置服务器状态 |

### Python 客户端 API (端口 5001-5003)

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/status` | GET | 获取客户端状态 |
| `/api/train` | POST | 开始训练 |
| `/api/stop` | POST | 停止训练 |
| `/api/losses` | GET | 获取 Loss 历史 |
| `/api/evaluate` | GET | 评估模型 |
| `/api/reset` | POST | 重置客户端 |
| `/api/weights` | GET | 获取当前模型权重 |

## 核心组件说明

### 1. C++ 聚合服务器

**文件结构：**
- `CMakeLists.txt` - CMake 构建配置
- `FedAvgServer.h/cpp` - FedAvg 聚合算法实现
- `main.cpp` - HTTP 服务器入口

**FedAvg 算法实现：**
```cpp
// 加权平均计算
double weight = static_cast<double>(sample_count) / total_samples;
result += weight * tensor;
```

### 2. Python GNN 客户端

**文件结构：**
- `requirements.txt` - Python 依赖
- `config.py` - 配置文件
- `gnn_model.py` - GCN 图卷积网络
- `data_loader.py` - 联邦数据划分 (支持 IID/Non-IID)
- `trainer.py` - 联邦训练器
- `client_api_server.py` - Flask API 服务器
- `start_clients.py` - 多客户端启动脚本

**数据划分策略：**
- **IID**：随机均匀划分数据
- **Non-IID**：按类别划分，每个客户端只拥有部分类别的数据

### 3. Vue 3 前端

**文件结构：**
- `package.json` - 依赖配置
- `vite.config.js` - Vite 构建配置
- `src/`
  - `main.js` - 应用入口
  - `App.vue` - 主组件
  - `style.css` - 全局样式
  - `stores/clients.js` - Pinia 状态管理
  - `components/`
    - `ControlPanel.vue` - 训练控制面板
    - `ClientStatus.vue` - 客户端状态显示
    - `LossChart.vue` - Loss 曲线实时绘制

**主要功能：**
- 实时轮询客户端状态 (默认 2 秒)
- 配置学习率、训练轮数
- 下发训练/停止指令
- 使用 Chart.js 实时绘制 Loss 曲线

## 训练流程

```
1. 前端 → 客户端: POST /api/train { learning_rate, rounds }
2. 客户端: 本地训练 num_epochs
3. 客户端 → 聚合服务器: POST /api/submit { weights, sample_count }
4. 聚合服务器: 等待所有客户端提交
5. 手动/自动触发聚合: FedAvg 加权平均
6. 客户端 → 聚合服务器: GET /api/weights (下载全局权重)
7. 客户端: 用全局权重更新本地模型
8. 重复步骤 2-7，直到完成所有 rounds
```

## 技术栈

| 组件 | 技术 |
|------|------|
| C++ 服务器 | C++17, cpp-httplib, nlohmann/json, CMake |
| Python 客户端 | Python 3.8+, PyTorch, PyTorch Geometric, Flask |
| Vue 3 前端 | Vue 3, Vite, Pinia, Chart.js, vue-chartjs, Axios |

## 注意事项

1. **首次运行**：Python 客户端会自动下载 Cora 数据集 (~100MB)
2. **GPU 支持**：PyTorch 会自动检测并使用 CUDA，否则使用 CPU
3. **端口冲突**：确保 8080, 5001, 5002, 5003, 3000 端口未被占用
4. **防火墙**：如果在多机环境运行，需要开放相应端口

## 扩展建议

1. **添加更多数据集**：在 `data_loader.py` 中支持更多图数据集
2. **增加客户端数量**：修改 `config.py` 中的 `NUM_CLIENTS`
3. **实现其他聚合算法**：在 `FedAvgServer.cpp` 中添加 FedProx, FedAdam 等
4. **添加安全性**：实现加密权重传输、差分隐私等
5. **支持异步训练**：实现异步联邦学习模式
