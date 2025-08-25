# Gemma-3-1b Tool Use 微调项目

基于第一性原理的Gemma-3-1b工具调用微调项目，让模型学会使用工具。

## 🎯 项目目标

- 微调Gemma-3-1b模型以支持工具调用
- 使用真实的工具调用数据集进行训练
- 基于LoRA技术进行高效微调
- **使用uv进行依赖管理**

## 📦 依赖管理

本项目使用 `uv` 进行依赖管理：

```bash
# 安装依赖
uv sync

# 运行训练脚本
uv run python simple_train.py
```

## 🚀 快速开始

### 本地训练

1. 克隆项目
```bash
git clone https://github.com/layue13/ft.git
cd ft
```

2. 安装依赖
```bash
uv sync
```

3. 运行训练
```bash
uv run python simple_train.py
```

### 云端训练 (HF Jobs)

#### 硬件选择

根据[Hugging Face Jobs文档](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli#hf-jobs)，有多种硬件选择：

**经济型选择** (推荐):
```bash
# T4 GPU - 性价比最高
hf jobs run --flavor t4-small --secrets HF_TOKEN pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel bash -c "
pip install uv &&
git clone https://github.com/layue13/ft.git &&
cd ft &&
uv sync &&
uv run python hf_jobs_train.py
"

# L4 GPU - 中等性能
hf jobs run --flavor l4x1 --secrets HF_TOKEN pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel bash -c "
pip install uv &&
git clone https://github.com/layue13/ft.git &&
cd ft &&
uv sync &&
uv run python hf_jobs_train.py
"
```

**高性能选择**:
```bash
# A10G GPU - 平衡性能
hf jobs run --flavor a10g-small --secrets HF_TOKEN pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel bash -c "
pip install uv &&
git clone https://github.com/layue13/ft.git &&
cd ft &&
uv sync &&
uv run python hf_jobs_train.py
"

# A100 GPU - 最高性能
hf jobs run --flavor a100-large --secrets HF_TOKEN pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel bash -c "
pip install uv &&
git clone https://github.com/layue13/ft.git &&
cd ft &&
uv sync &&
uv run python hf_jobs_train.py
"
```

**CPU选择** (最经济):
```bash
# CPU训练 - 最便宜但较慢
hf jobs run --flavor cpu-upgrade --secrets HF_TOKEN pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel bash -c "
pip install uv &&
git clone https://github.com/layue13/ft.git &&
cd ft &&
uv sync &&
uv run python hf_jobs_train.py
"
```

#### 硬件选择指南

| 硬件 | 适用场景 | 训练时间 | 成本 | 推荐度 |
|------|----------|----------|------|--------|
| **T4-small** | 预算有限，不着急 | 2-4小时 | $0.5-1 | ⭐⭐⭐⭐⭐ |
| **L4x1** | 平衡性能和成本 | 1.5-3小时 | $1-1.5 | ⭐⭐⭐⭐ |
| **A10G-small** | 快速训练 | 1-2小时 | $1-2 | ⭐⭐⭐ |
| **A100-large** | 最快训练 | 30-60分钟 | $2-4 | ⭐⭐ |
| **CPU-upgrade** | 最经济 | 4-8小时 | $0.2-0.5 | ⭐⭐⭐ |

**推荐**: 首次尝试建议使用 `t4-small`，性价比最高！

## 📊 项目特性

| 特性 | 说明 |
|------|------|
| **模型** | google/gemma-3-1b-it |
| **数据集** | shawhin/tool-use-finetuning (477个样本) |
| **微调方法** | LoRA (Low-Rank Adaptation) |
| **依赖管理** | uv |
| **训练时间** | 2-4小时 (T4) / 1-2小时 (A10G) / 30-60分钟 (A100) |
| **预期成本** | $0.5-1 (T4) / $1-2 (A10G) / $2-4 (A100) |

## 🔧 技术栈

- **模型**: Gemma-3-1b-it
- **微调方法**: LoRA (r=16, alpha=32)
- **框架**: Transformers + PEFT
- **数据格式**: Gemma对话格式
- **优化器**: AdamW (lr=2e-5)
- **精度**: bfloat16 (GPU) / float32 (CPU)
- **依赖管理**: uv

## 📁 项目结构

```
ft/
├── simple_train.py              # 🏠 本地训练脚本
├── hf_jobs_train.py            # ☁️ HF Jobs训练脚本
├── pyproject.toml              # 📦 uv依赖配置
├── README.md                   # 📖 项目说明
└── .gitignore                  # 🚫 Git忽略文件
```

## 🎉 获取成果

训练完成后，你将获得：

1. **本地模型**: `./gemma3-tool-use/`
2. **Hub模型**: `your-username/gemma3-1b-tool-use`
3. **训练日志**: Weights & Biases记录

### 使用微调后的模型

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载微调后的模型
model = AutoModelForCausalLM.from_pretrained("your-username/gemma3-1b-tool-use")
tokenizer = AutoTokenizer.from_pretrained("your-username/gemma3-1b-tool-use")

# 进行工具调用推理
prompt = "What's the weather like in Beijing?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## 🚀 uv的优势

- **快速安装**: 比pip快10-100倍
- **依赖解析**: 更智能的依赖冲突解决
- **虚拟环境**: 自动管理虚拟环境
- **缓存优化**: 智能缓存减少重复下载
- **跨平台**: 支持Windows、macOS、Linux

## 📝 许可证

本项目遵循MIT许可证。

## 🤝 贡献

欢迎提交Issue和Pull Request！