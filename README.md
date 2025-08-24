# Gemma3-1b 工具调用微调项目

使用PEFT（Parameter Efficient Fine-Tuning）微调Gemma3-1b模型，使其支持工具调用功能。

## 项目概述

本项目使用shawhin/tool-use-finetuning数据集对Google的Gemma3-1b模型进行微调，通过LoRA（Low-Rank Adaptation）方法实现参数高效的微调，使模型能够理解和执行工具调用任务。

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

### 中国网络环境 (推荐)

#### 1. 测试镜像站连接

```cmd
python scripts/test_mirrors.py
```

#### 2. 一键训练 (自动选择最佳镜像站)

```cmd
scripts/train_china.bat
```

#### 3. 手动指定镜像站训练

```cmd
# 使用HF Mirror标准镜像站 (推荐)
uv run python scripts/train_with_mirror.py --mirror hf_mirror

# 使用ModelScope镜像站
uv run python scripts/train_with_mirror.py --mirror modelscope

# 使用清华镜像站
uv run python scripts/train_with_mirror.py --mirror tsinghua

# 自动检测最佳镜像站
uv run python scripts/train_with_mirror.py --mirror auto

### RTX 4090环境

#### 一键训练 (RTX 4090优化)

```cmd
scripts/train_rtx4090.bat
```

#### 手动训练 (RTX 4090优化)

```cmd
uv run python scripts/train_with_mirror.py --config configs/training_config_rtx4090.yaml --mirror hf_mirror
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

### 中国网络环境配置

中国网络环境使用`configs/training_config_china.yaml`，主要特性：

 - 自动镜像站检测和选择
 - 支持HF Mirror、ModelScope、清华镜像等国内镜像站
 - 优化的网络连接参数
 - 减少数据集大小以适应网络限制

### RTX 4090优化配置

RTX 4090使用`configs/training_config_rtx4090.yaml`，主要优化：

 - 更大的batch size（8）充分利用24GB显存
 - 更长的序列长度（2048）提高训练效果
 - 更多的数据集样本（1000）提升模型性能
 - 更高的LoRA rank（16）增强模型表达能力
 - 关闭梯度检查点以提升训练速度
 - 启用pin_memory和更多worker提升数据加载效率

## 技术栈

- **模型**: Google Gemma3-1b
- **微调方法**: PEFT LoRA
- **数据集**: shawhin/tool-use-finetuning
- **框架**: Transformers, PyTorch
- **包管理**: UV

## 许可证

MIT License
