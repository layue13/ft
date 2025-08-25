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
apt-get update && apt-get install -y git &&
pip install uv &&
git clone https://github.com/layue13/ft.git &&
cd ft &&
uv sync &&
uv run python hf_jobs_train.py
"

# L4 GPU - 中等性能
hf jobs run --flavor l4x1 --secrets HF_TOKEN pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel bash -c "
apt-get update && apt-get install -y git &&
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
apt-get update && apt-get install -y git &&
pip install uv &&
git clone https://github.com/layue13/ft.git &&
cd ft &&
uv sync &&
uv run python hf_jobs_train.py
"

# A100 GPU - 最高性能
hf jobs run --flavor a100-large --secrets HF_TOKEN pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel bash -c "
apt-get update && apt-get install -y git &&
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
apt-get update && apt-get install -y git &&
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
├── merge_lora.py                # 🔄 LoRA权重合并脚本
├── convert_to_gguf.py           # 🚀 GGUF格式转换脚本
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

#### 方法1: Python代码使用

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

#### 方法2: LM Studio使用 (推荐)

**步骤1: 下载模型**
```bash
# 从Hugging Face Hub下载模型
git lfs install
git clone https://huggingface.co/your-username/gemma3-1b-tool-use
```

**步骤2: 在LM Studio中加载**
1. 打开LM Studio
2. 点击 "Local Server" 标签
3. 点击 "Browse" 选择模型文件夹 (`gemma3-1b-tool-use`)
4. 点击 "Load Model"

**步骤3: 配置聊天界面**
1. 切换到 "Chat" 标签
2. 设置合适的参数：
   - **Temperature**: 0.7-0.9 (创造性)
   - **Top P**: 0.9
   - **Max Tokens**: 512
   - **Stop Sequences**: `</s>`, `<eos>`

**步骤4: 工具调用示例**
```
用户: 帮我查询北京的天气

助手: <tool_call>
{
 "tool_name": "weather",
 "args": {
   "location": "Beijing"
 }
}
</tool_call>
```

**步骤5: 高级配置**
- **Context Length**: 4096 (Gemma-3-1b支持)
- **GPU Layers**: 根据你的GPU内存调整
- **Threads**: CPU核心数

#### 方法3: MLX优化 (Apple Silicon)

对于Apple Silicon Mac，可以使用MLX获得最佳性能：

```bash
# 安装MLX
pip install mlx

# 使用MLX加载模型
import mlx.core as mx
from transformers import AutoTokenizer

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained("your-username/gemma3-1b-tool-use")

# 使用MLX进行推理 (需要MLX适配)
# 注意: 需要将模型转换为MLX格式
```

#### 方法4: 转换为GGUF格式 (推荐)

转换为GGUF格式获得最佳性能和兼容性：

```bash
# 方法1: 使用llama.cpp转换
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make

# 转换模型 (需要先合并LoRA权重)
python convert.py your-username/gemma3-1b-tool-use \
    --outfile gemma3-1b-tool-use.gguf \
    --outtype q4_k_m

# 方法2: 使用transformers-to-gguf
pip install transformers-to-gguf
transformers-to-gguf your-username/gemma3-1b-tool-use \
    --output gemma3-1b-tool-use.gguf \
    --quantize q4_k_m
```

**GGUF优势**:
- 🚀 **更快推理**: 比原格式快2-5倍
- 💾 **更小体积**: 量化后体积减少50-75%
- 🔧 **更好兼容**: 支持更多推理框架
- 🖥️ **更低资源**: 可在CPU上高效运行

#### 方法5: 合并LoRA权重 (推荐)

为了获得最佳兼容性，建议先合并LoRA权重：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-it")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")

# 加载LoRA权重
model = PeftModel.from_pretrained(base_model, "your-username/gemma3-1b-tool-use")

# 合并权重
merged_model = model.merge_and_unload()

# 保存合并后的模型
merged_model.save_pretrained("./gemma3-1b-tool-use-merged")
tokenizer.save_pretrained("./gemma3-1b-tool-use-merged")

print("✅ LoRA权重已合并，模型已保存")
```

**合并后的优势**:
- ✅ **完全兼容**: 所有推理框架都支持
- 🚀 **更快加载**: 无需动态加载LoRA
- 💾 **更小体积**: 比分离存储更紧凑
- 🔧 **更好部署**: 适合生产环境

### 🚀 手动转换脚本

使用两个独立的脚本进行手动转换：

#### 步骤1: 合并LoRA权重

```bash
# 运行LoRA权重合并脚本
python merge_lora.py

# 按提示输入:
# - 模型名称 (例如: layue13/gemma3-1b-tool-use)
# - 输出目录 (默认: ./gemma3-1b-tool-use-merged)
```

#### 步骤2: 转换为GGUF格式

```bash
# 运行GGUF转换脚本
python convert_to_gguf.py

# 按提示输入:
# - 模型路径 (合并后的模型目录)
# - 输出文件 (默认: ./gemma3-1b-tool-use.gguf)
# - 量化类型 (推荐: q4_k_m)
```

**转换脚本功能**:
- 🔄 **独立合并**: 专门的LoRA权重合并
- 🚀 **GGUF转换**: 支持多种量化选项
- 📝 **自动生成**: 生成使用示例脚本
- 🔧 **依赖检查**: 自动检查和安装依赖

### 工具调用格式说明

训练后的模型支持以下格式：

```
<bos><start_of_turn>user
你的问题
<end_of_turn>
<start_of_turn>model
<tool_call>
{
 "tool_name": "工具名称",
 "args": {
   "参数1": "值1",
   "参数2": "值2"
 }
}
</tool_call>
<end_of_turn><eos>
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