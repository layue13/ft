# Gemma-3-1b Tool Use Fine-tuning

基于第一性原理的简化微调项目，训练Gemma-3-1b模型支持工具调用。

## 快速开始

### 本地运行
```bash
# 安装依赖
uv sync

# 开始训练
uv run simple_train.py
```

### HF Jobs云端训练

1. **设置Token**：
   ```bash
   # 获取token: https://huggingface.co/settings/tokens
   export HF_TOKEN=your_write_token_here
   ```

2. **提交训练任务**：
   ```bash
   uv run submit_hf_job.py
   ```

3. **监控训练状态**：
   ```bash
   uv run monitor_job.py <job_id>
   ```

## 项目管理

### 添加新依赖
```bash
uv add package_name
```

### 查看依赖
```bash
uv pip list
```

### 同步环境
```bash
uv sync
```

## 输出

训练完成后会生成：
- `./gemma3-tool-use/` - LoRA适配器
- `./gemma3-tool-use-merged/` - 完整合并模型
- 自动上传到HF Hub: `layue13/gemma3-tool-use`

## 配置

- **GPU**: A10G Small ($0.7/小时)
- **训练时间**: 15-20分钟
- **数据集**: shawhin/tool-use-finetuning (200样本)
- **方法**: LoRA微调 + 权重合并

## 文件说明

- `simple_train.py` - 主训练脚本
- `submit_hf_job.py` - HF Jobs任务提交
- `monitor_job.py` - 训练状态监控
- `pyproject.toml` - uv项目配置