# Gemma3-1b 工具调用微调项目 - 优化版本

使用PEFT（Parameter Efficient Fine-Tuning）微调Gemma3-1b模型，使其支持工具调用功能。

## 项目概述

本项目使用shawhin/tool-use-finetuning数据集对Google的Gemma3-1b模型进行微调，通过LoRA（Low-Rank Adaptation）方法实现参数高效的微调，使模型能够理解和执行工具调用任务。

### 🚀 最新优化

- **简化数据处理逻辑**：优化工具调用格式转换，提高处理效率
- **增强验证机制**：添加工具调用格式验证和配置验证
- **针对性评估指标**：新增工具调用准确率、F1分数等专业指标
- **优化配置管理**：简化配置文件，减少冗余参数
- **增强测试覆盖**：完善单元测试和集成测试
- **性能优化**：提供优化版配置文件，提升训练效率

## 环境要求

- Python >= 3.9, < 3.14 (推荐Python 3.11.6或3.12.0)
- CUDA兼容的GPU（推荐16GB+显存）
- UV包管理器
- Hugging Face账号（需要申请Gemma3-1b-it模型访问权限）

> **注意**: 如果遇到Python版本问题，请参考 [WINDOWS_SETUP.md](WINDOWS_SETUP.md) 获取详细解决方案。
> 
> **中国用户**: 如果遇到网络连接问题，请使用 `scripts/train_china.bat` 脚本，它会自动选择最佳镜像站。

## 安装

1. 克隆项目：
```bash
git clone <your-repo-url>
cd gemma3-tool-finetuning
```

2. 使用UV安装依赖：
```bash
uv sync
```

## 项目结构

```
gemma3-tool-finetuning/
├── src/
│   ├── __init__.py
│   ├── data_processor.py      # 数据处理模块（优化版）
│   ├── model_config.py        # 模型配置
│   ├── trainer.py             # 训练器（优化版）
│   ├── utils.py               # 工具函数（优化版）
│   └── mirror_utils.py        # 镜像站工具
├── configs/
│   ├── training_config.yaml           # 标准训练配置
│   ├── training_config_optimized.yaml # 优化训练配置
│   └── training_config_*.yaml         # 其他环境配置
├── scripts/
│   ├── prepare_data.py        # 数据准备脚本
│   ├── train.py               # 训练脚本（优化版）
│   ├── evaluate.py            # 评估脚本（优化版）
│   └── train_*.bat            # 环境特定脚本
├── tests/                     # 测试文件（增强版）
├── pyproject.toml            # 项目配置
└── README.md                 # 项目文档
```

## 使用方法

### 快速开始（推荐）

#### 1. 使用优化配置训练

```bash
# 使用优化配置进行训练
uv run python scripts/train.py --config configs/training_config_optimized.yaml
```

#### 2. 评估模型性能

```bash
# 评估模型工具调用能力
uv run python scripts/evaluate.py --model_path ./outputs --max_samples 100
```

### 标准训练流程

#### 1. 数据准备

```bash
uv run python scripts/prepare_data.py
```

#### 2. 开始训练

```bash
uv run python scripts/train.py --config configs/training_config.yaml
```

#### 3. 评估模型

```bash
uv run python scripts/evaluate.py --model_path ./outputs/checkpoint-final
```

### 环境特定训练

#### Windows CUDA环境

```cmd
scripts/setup_windows.bat
scripts/train_windows.bat
```

#### 中国网络环境

```cmd
scripts/train_china.bat
```

#### RTX 4090环境

```cmd
scripts/train_rtx4090.bat
```

## 配置说明

### 优化配置特性

**training_config_optimized.yaml** 包含以下优化：

- **数据集限制**：限制为1000样本，提高训练效率
- **序列长度优化**：减少到1024，降低内存占用
- **学习率调整**：降低到1e-4，提高训练稳定性
- **训练轮数优化**：减少到2轮，避免过拟合
- **评估间隔优化**：增加评估间隔，减少计算开销
- **内存优化**：启用pin_memory，提升数据加载效率

### 主要配置参数

- `model.name`: 基础模型名称
- `dataset.max_samples`: 数据集大小限制
- `lora.r`: LoRA rank参数
- `training.learning_rate`: 学习率
- `training.num_train_epochs`: 训练轮数
- `data_processing.max_seq_length`: 最大序列长度

## 评估指标

### 新增工具调用指标

- **tool_call_accuracy**: 工具调用准确率
- **tool_name_accuracy**: 工具名称准确率
- **tool_args_accuracy**: 工具参数准确率
- **tool_call_f1**: 工具调用F1分数
- **exact_match**: 完全匹配率

### 标准指标

- **eval_loss**: 验证损失
- **eval_accuracy**: 验证准确率

## 技术栈

- **模型**: Google Gemma3-1b
- **微调方法**: PEFT LoRA
- **数据集**: shawhin/tool-use-finetuning
- **框架**: Transformers, PyTorch
- **包管理**: UV

## 测试

运行测试套件：

```bash
uv run pytest tests/ -v
```

测试覆盖：
- 工具调用格式验证
- 数据处理逻辑
- 配置验证
- 端到端集成测试

## 性能优化建议

1. **使用优化配置**：优先使用 `training_config_optimized.yaml`
2. **调整数据集大小**：根据显存限制调整 `max_samples`
3. **优化序列长度**：根据任务需求调整 `max_seq_length`
4. **监控资源使用**：使用 `log_system_info()` 监控系统状态

## 许可证

MIT License
