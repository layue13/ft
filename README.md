# Gemma3-1b 工具调用微调项目

使用PEFT（Parameter Efficient Fine-Tuning）微调Gemma3-1b模型，使其支持工具调用功能。

## 项目概述

本项目使用shawhin/tool-use-finetuning数据集对Google的Gemma3-1b-it模型进行微调，通过LoRA（Low-Rank Adaptation）方法实现参数高效的微调，使模型能够理解和执行工具调用任务。

## 🚀 加速方案

### 1. Hugging Face Jobs (推荐)
使用Hugging Face Jobs进行云端微调，无需本地GPU：

```bash
# 安装依赖
pip install huggingface_hub

# 设置环境变量
export HF_TOKEN="your_huggingface_token"

# 运行微调
python scripts/train_hf_jobs.py --flavor a10g-small
```

**硬件选择**:
- `t4-small`: $0.40/小时 (16GB GPU)
- `a10g-small`: $1.00/小时 (24GB GPU) 
- `a10g-large`: $1.50/小时 (24GB GPU)
- `a100-large`: $4.00/小时 (80GB GPU)

### 2. Hugging Face Spaces
创建Space进行微调：

1. 将此项目推送到Hugging Face Hub
2. 创建新的Space，选择Gradio模板
3. 升级到GPU硬件
4. 运行微调

### 3. 本地训练
如果您有强大的本地GPU：

```bash
# 安装依赖
uv sync

# 运行训练
python scripts/train.py
```

## 环境要求

- Python >= 3.9
- CUDA兼容的GPU（推荐16GB+显存）
- UV包管理器

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
│   ├── data_processor.py      # 数据处理模块
│   ├── model_config.py        # 模型配置
│   ├── trainer.py             # 训练器
│   └── utils.py               # 工具函数
├── configs/
│   └── training_config.yaml   # 训练配置
├── scripts/
│   ├── prepare_data.py        # 数据准备脚本
│   ├── train.py               # 训练脚本
│   ├── train_hf_jobs.py       # HF Jobs训练脚本
│   └── evaluate.py            # 评估脚本
├── tests/                     # 测试文件
├── app.py                     # Space应用入口
├── pyproject.toml            # 项目配置
└── README.md                 # 项目文档
```

## 使用方法

### 🚀 快速启动

```bash
# 本地训练
python scripts/quick_start.py --mode local

# Hugging Face Jobs训练
export HF_TOKEN="your_token"
python scripts/quick_start.py --mode hf-jobs --flavor a10g-small

# Hugging Face Spaces训练
python scripts/quick_start.py --mode hf-spaces
```

### 详细步骤

#### 1. 数据准备

```bash
python scripts/prepare_data.py
```

#### 2. 开始训练

##### 本地训练
```bash
python scripts/train.py --config configs/training_config.yaml
```

##### Hugging Face Jobs训练
```bash
# 设置环境变量
export HF_TOKEN="your_huggingface_token"

# 运行微调
python scripts/train_hf_jobs.py --flavor a10g-small
```

##### Space训练
1. 部署到Hugging Face Spaces
2. 升级到GPU硬件
3. 通过Web界面启动训练

### 3. 评估模型

```bash
python scripts/evaluate.py --model_path ./outputs/checkpoint-final
```

## 配置说明

主要配置参数在`configs/training_config.yaml`中：

- `model_name`: 基础模型名称（google/gemma-3-1b-it）
- `dataset_name`: 数据集名称（shawhin/tool-use-finetuning）
- `lora_config`: LoRA配置参数
- `training_args`: 训练参数

## 技术栈

- **模型**: Google Gemma3-1b-it
- **微调方法**: PEFT LoRA
- **数据集**: shawhin/tool-use-finetuning
- **框架**: Transformers, PyTorch
- **包管理**: UV
- **加速平台**: Hugging Face Jobs/Spaces

## 成本估算

使用Hugging Face Jobs的预估成本：

| 硬件配置 | 每小时价格 | 3小时训练 | 10小时训练 |
|---------|-----------|-----------|------------|
| T4 Small | $0.40 | $1.20 | $4.00 |
| A10G Small | $1.00 | $3.00 | $10.00 |
| A10G Large | $1.50 | $4.50 | $15.00 |
| A100 Large | $4.00 | $12.00 | $40.00 |

## 许可证

MIT License
