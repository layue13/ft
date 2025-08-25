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
# 方式1: 单行命令（推荐，避免换行问题）
uv run hf jobs run --flavor a10g-small --secrets HF_TOKEN pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel -- bash -c "apt-get update && apt-get install -y git && git clone https://github.com/layue13/ft.git && cd ft && pip install uv && pip install 'transformers==4.44.2' 'accelerate==0.33.0' datasets peft torch && uv run python hf_jobs_train.py"

# 方式2: 多行命令（如果终端支持）
uv run hf jobs run \
    --flavor a10g-small \
    --secrets HF_TOKEN \
    pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
    -- bash -c "apt-get update && apt-get install -y git && git clone https://github.com/layue13/ft.git && cd ft && pip install uv && pip install 'transformers==4.44.2' 'accelerate==0.33.0' datasets peft torch && uv run python hf_jobs_train.py"

# 🚀 最佳选择：使用HF Jobs的uv支持
uv run hf jobs uv --flavor a10g-small \
    --secrets HF_TOKEN \
    --script "
    # /// script
    # dependencies = [
    #     'transformers>=4.40.0',
    #     'datasets>=2.14.0', 
    #     'peft>=0.7.0',
    #     'accelerate>=0.20.0',
    #     'torch>=2.2.0'
    # ]
    # ///
    
    import subprocess
    import os
    
    # Install git if not available
    if os.system('which git') != 0:
        subprocess.run(['apt-get', 'update'], check=True)
        subprocess.run(['apt-get', 'install', '-y', 'git'], check=True)
    
    subprocess.run(['git', 'clone', 'https://github.com/layue13/ft.git'], check=True)
    subprocess.run(['python', 'ft/hf_jobs_train.py'], check=True)
    "

# 方式3: 传统单行方式（确保正确格式）
uv run hf jobs run --flavor a10g-small --secrets HF_TOKEN pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel -- bash -c "apt-get update && apt-get install -y git && git clone https://github.com/layue13/ft.git && cd ft && pip install uv && pip install 'transformers==4.44.2' 'accelerate==0.33.0' datasets peft torch && uv run python hf_jobs_train.py"
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
   **解决方案**: 使用单行命令格式:
   ```bash
   uv run hf jobs run --flavor a10g-small --secrets HF_TOKEN pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel -- bash -c "apt-get update && apt-get install -y git && git clone https://github.com/layue13/ft.git && cd ft && pip install uv && pip install 'transformers==4.44.2' 'accelerate==0.33.0' datasets peft torch && uv run python hf_jobs_train.py"
   ```

1c. **库版本兼容性错误** 🔧
   ```
   TypeError: Accelerator.unwrap_model() got an unexpected keyword argument 'keep_torch_compile'
   ```
   **原因**: transformers和accelerate版本不兼容
   **解决方案**: 使用具体的兼容版本:
   - transformers==4.44.2
   - accelerate==0.33.0
   
   **更新后的命令**: 见上方命令示例

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