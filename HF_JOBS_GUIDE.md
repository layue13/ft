# Hugging Face Jobs 云端微调指南

## 🚀 什么是HF Jobs？

HF Jobs是Hugging Face的云端计算平台，让你可以在强大的GPU上训练模型而无需本地硬件。

## 📋 前置条件

1. **Pro账户**：需要HF Pro订阅（$9/月）
2. **CLI工具**：最新版本的huggingface-cli

## 🛠️ 快速开始

### 1. 设置环境

```bash
# 升级huggingface-hub (已完成)
uv add "huggingface_hub[cli]"

# 登录你的Pro账户
uv run hf auth login

# 检查登录状态
uv run hf auth whoami

# 查看可用任务
uv run hf jobs ps
```

### 2. 查看可用硬件

```bash
# CPU选项
- cpu-basic: 基础CPU
- cpu-upgrade: 高性能CPU

# GPU选项  
- t4-small: NVIDIA T4 (16GB)
- a10g-small: NVIDIA A10G (24GB)
- a100-large: NVIDIA A100 (80GB)
```

### 3. 提交训练任务

```bash
# 正确的命令格式（本地运行，云端执行）
# 方式1: 使用项目的模块化架构（推荐）
uv run hf jobs run --flavor a10g-small --secrets HF_TOKEN pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel -- bash -c "apt-get update && apt-get install -y git && git clone https://github.com/layue13/ft.git && cd ft && pip install uv && uv sync && uv run python scripts/train.py --config configs/training_config_public.yaml"

# 方式2: 多行命令（如果终端支持）
uv run hf jobs run \
    --flavor a10g-small \
    --secrets HF_TOKEN \
    pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
    -- bash -c "apt-get update && apt-get install -y git && git clone https://github.com/layue13/ft.git && cd ft && pip install uv && uv sync && uv run python scripts/train.py --config configs/training_config_public.yaml"

# 🚀 **第一性原理：最简方案**

## 方案A：内联UV脚本（推荐）
```bash
uv run hf jobs uv --flavor a10g-small --secrets HF_TOKEN --script "
# /// script
# dependencies = ['transformers', 'datasets', 'peft', 'torch', 'accelerate']
# ///

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

# Gemma-3-1b Tool Use 微调
tokenizer = AutoTokenizer.from_pretrained('google/gemma2-1.1b-it', padding_side='right')
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained('google/gemma2-1.1b-it', torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True)

# LoRA - 针对Gemma优化
lora_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.1, task_type=TaskType.CAUSAL_LM, target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Tool Use数据格式化 - 正确格式
def format_tool_use(example):
    if 'trace' not in example or not example.get('tool_needed'): 
        return {'text': '<bos><start_of_turn>user\\nhello<end_of_turn>\\n<start_of_turn>model\\nHello!<end_of_turn><eos>'}
    conv = '<bos>'
    for msg in example['trace']:
        if msg.get('role') == 'user': conv += f'<start_of_turn>user\\n{msg.get(\"content\", \"\")}<end_of_turn>\\n'
    if example.get('tool_needed') and example.get('tool_name'):
        tool_call = f'<tool_call>\\n{{\\n \"tool_name\": \"{example[\"tool_name\"]}\",\\n \"args\": {{}}\\n}}\\n</tool_call>'
        conv += f'<start_of_turn>model\\n{tool_call}<end_of_turn>'
    return {'text': conv + '<eos>'}

dataset = load_dataset('shawhin/tool-use-finetuning', split='train[:200]').map(format_tool_use)
tokenized = dataset.map(lambda x: tokenizer(x['text'], truncation=True, padding='max_length', max_length=512), batched=True, remove_columns=dataset.column_names)
tokenized = tokenized.add_column('labels', tokenized['input_ids'])

# 训练 - Tool Use优化参数
trainer = Trainer(
    model=model,
    args=TrainingArguments(output_dir='./gemma3-tool-use', num_train_epochs=2, per_device_train_batch_size=1, gradient_accumulation_steps=4, learning_rate=2e-5, warmup_ratio=0.1, logging_steps=10, save_strategy='epoch', push_to_hub=True, hub_model_id='gemma3-1b-tool-use', bf16=True, gradient_checkpointing=True, remove_unused_columns=False),
    train_dataset=tokenized,
    tokenizer=tokenizer
)
trainer.train()
print('🎉 训练完成！')
"
```

## 方案B：使用极简脚本
```bash
uv run hf jobs run --flavor a10g-small --secrets HF_TOKEN pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel -- bash -c "git clone https://github.com/layue13/ft.git && cd ft && python simple_train.py"
```

## 🎯 **Gemma-3-1b Tool Use 优势**

✅ **专业目标**：
- 🤖 Gemma-3-1b: 优秀的小模型
- 🛠 Tool Use: 工具调用能力
- 📊 shawhin/tool-use-finetuning: 专业数据集
- 🎯 XML格式: <tool_call>标准

✅ **优化设置**：
- LoRA r=16: 适合工具调用的复杂性
- 目标模块: Gemma全注意力层
- 序列长度512: 支持复杂工具调用
- 学习率2e-5: 保护预训练知识

✅ **云端训练**：
- 200训练样本: 快速验证
- 2个epoch: 充分学习
- A10G GPU: 高效训练
- 15-20分钟完成

💰 **成本估算**：15-20分钟训练 ≈ $0.50
```

### 4. 监控任务

```bash
# 查看所有任务
uv run hf jobs ps

# 查看特定任务详情
uv run hf jobs inspect <job-id>

# 查看任务日志
uv run hf jobs logs <job-id>

# 取消任务
uv run hf jobs cancel <job-id>
```

## 💰 费用估算

| 硬件类型 | 每小时费用 | 适用场景 |
|----------|------------|----------|
| t4-small | ~$0.50 | 小模型微调 |
| a10g-small | ~$1.50 | 中等模型微调 |  
| a100-large | ~$4.00 | 大模型训练 |

**对于Gemma-3-1b微调，推荐使用 `a10g-small`，预估成本：**
- 2小时训练 ≈ $3
- 比购买GPU便宜得多！

## 📝 使用我们的训练脚本

项目中已包含 `hf_jobs_train.py`，这是专门为HF Jobs优化的训练脚本：

```bash
# 提交我们的微调任务
hf jobs run --flavor a10g-small \
    --env HUGGINGFACE_HUB_TOKEN=$HF_TOKEN \
    pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
    bash -c "
        git clone https://github.com/layue13/ft.git && 
        cd ft && 
        pip install transformers datasets peft accelerate torch && 
        python hf_jobs_train.py
    "
```

## ✅ 优势

1. **无硬件需求**：无需购买昂贵GPU
2. **弹性扩展**：按需选择硬件规格
3. **自动保存**：训练完成后模型自动上传到你的Hub
4. **专业环境**：预配置的PyTorch环境
5. **按需付费**：只为实际使用时间付费

## 🔧 故障排除

### 常见问题

1. **git命令未找到** 🔧
   ```
   bash: line 1: git: command not found
   ```
   **解决方案**: 使用正确的命令格式（注意 `--` 分隔符）:
   ```bash
   uv run hf jobs run --flavor a10g-small --secrets HF_TOKEN pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
       -- bash -c "apt-get update && apt-get install -y git && git clone https://github.com/layue13/ft.git && cd ft && pip install uv && uv run python hf_jobs_train.py"
   ```

1b. **命令解析错误** 🔧
   ```
   usage: hf <command> [<args>] jobs run: error: the following arguments are required: image
   zsh: no such file or directory: pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
   ```
   **原因**: 多行命令在某些终端中被错误解析
   **解决方案**: 使用项目的模块化架构:
   ```bash
   uv run hf jobs run --flavor a10g-small --secrets HF_TOKEN pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel -- bash -c "apt-get update && apt-get install -y git && git clone https://github.com/layue13/ft.git && cd ft && pip install uv && uv sync && uv run python scripts/train.py --config configs/training_config_public.yaml"
   ```

1c. **库版本兼容性错误** 🔧
   ```
   TypeError: Accelerator.unwrap_model() got an unexpected keyword argument 'keep_torch_compile'
   ```
   **根本原因**: 使用了独立的hf_jobs_train.py脚本而不是项目的模块化架构
   **解决方案**: 
   - ✅ 使用项目的scripts/train.py（已通过本地测试）
   - ✅ 使用uv sync自动管理依赖版本
   - ✅ 使用配置文件而不是硬编码参数
   
   **修复后的命令**: 见上方新命令格式

2. **认证失败**
   ```bash
   huggingface-cli login --token <your-token>
   ```

3. **任务失败**
   ```bash
   # 查看详细日志
   uv run hf jobs logs <job-id>
   ```

4. **内存不足**
   - 升级到更大的GPU规格
   - 减少batch_size
   - 启用gradient_checkpointing

### 调试技巧

```bash
# 测试脚本（不启动实际训练）
hf jobs run --flavor cpu-basic python:3.11 \
    python -c "import torch; print('环境测试成功')"
```

## 🎯 下一步

1. 获取HF Pro账户
2. 克隆这个仓库到GitHub（已完成）
3. 使用上述命令提交你的微调任务
4. 监控训练进度
5. 训练完成后在HF Hub下载你的微调模型

**开始你的云端微调之旅吧！** 🚀