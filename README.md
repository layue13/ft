# Gemma3-1b 工具调用微调项目

使用PEFT（Parameter Efficient Fine-Tuning）微调Gemma3-1b模型，使其支持工具调用功能。

## 项目概述

本项目使用shawhin/tool-use-finetuning数据集对Google的Gemma3-1b模型进行微调，通过LoRA（Low-Rank Adaptation）方法实现参数高效的微调，使模型能够理解和执行工具调用任务。

## 环境要求

- Python >= 3.9
- CUDA兼容的GPU（推荐16GB+显存）
- UV包管理器
- Hugging Face账号（需要申请Gemma3-1b-it模型访问权限）

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
│   └── evaluate.py            # 评估脚本
├── tests/                     # 测试文件
├── pyproject.toml            # 项目配置
└── README.md                 # 项目文档
```

## 使用方法

### Linux/macOS环境

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

### Windows CUDA环境

#### 1. 环境设置

```cmd
scripts/setup_windows.bat
```

#### 2. 登录Hugging Face

```cmd
huggingface-cli login
```

#### 3. 数据准备

```cmd
uv run python scripts/prepare_data.py
```

#### 4. 开始训练

```cmd
scripts/train_windows.bat
```

或者手动运行：

```cmd
uv run python scripts/train.py --config configs/training_config_windows.yaml
```

#### 5. 评估模型

```cmd
uv run python scripts/evaluate.py --model_path ./outputs/checkpoint-final
```

## 配置说明

主要配置参数在`configs/training_config.yaml`中：

- `model_name`: 基础模型名称（google/gemma-3-1b-it）
- `dataset_name`: 数据集名称（shawhin/tool-use-finetuning）
- `lora_config`: LoRA配置参数
- `training_args`: 训练参数

### Windows优化配置

Windows环境使用`configs/training_config_windows.yaml`，主要优化：

- 较小的batch size（2）以适应Windows内存限制
- 增加梯度累积步数（8）以保持有效batch size
- 减少数据加载worker数量（2）
- 启用梯度检查点以节省内存

## 技术栈

- **模型**: Google Gemma3-1b
- **微调方法**: PEFT LoRA
- **数据集**: shawhin/tool-use-finetuning
- **框架**: Transformers, PyTorch
- **包管理**: UV

## 许可证

MIT License
