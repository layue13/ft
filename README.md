# 🚀 极简AI微调项目

**第一性原理**：用最少的代码，最简的方式，完成AI模型微调。

## 🎯 核心理念

- ✨ **极简至上**：一个文件包含所有训练逻辑
- 🚀 **开箱即用**：无复杂配置，无多层架构
- ☁️ **云端优先**：直接在HF Jobs上运行
- 💰 **成本友好**：2-3分钟训练，成本 ≈ $0.10

## ⚡ 两种运行方式

### 方式A：内联脚本（推荐）
```bash
uv run hf jobs uv --flavor a10g-small --secrets HF_TOKEN --script "
# /// script  
# dependencies = ['transformers', 'datasets', 'peft', 'torch', 'accelerate']
# ///
# 完整训练代码内联在这里
"
```

### 方式B：极简脚本
```bash
uv run hf jobs run --flavor a10g-small --secrets HF_TOKEN pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel -- bash -c "git clone https://github.com/layue13/ft.git && cd ft && python simple_train.py"
```

详细命令见 `HF_JOBS_GUIDE.md`

## 📁 项目结构（极简版）

```
ft/
├── simple_train.py           # 🎯 核心：一个文件包含所有训练逻辑  
├── HF_JOBS_GUIDE.md         # ☁️ 云端训练指南
├── configs/training_config_public.yaml  # ⚙️ 基础配置
├── pyproject.toml           # 📦 依赖管理
└── README.md               # 📖 本文件
```

## 🔥 第一性原理优势

| 传统方式 | 🆚 | 极简方式 |
|---------|---|---------|
| 10+文件 | → | 1个文件 |
| 多层模块 | → | 直接逻辑 |
| 复杂配置 | → | 硬编码参数 |
| 本地环境 | → | 云端运行 |
| 高成本 | → | $0.10完成 |

## 🎯 微调目标

- **模型**：DialoGPT-small（公开可用）
- **方法**：LoRA微调（参数高效）
- **数据**：50样本快速验证
- **时长**：1个epoch，2-3分钟

## 🚀 立即开始

1. 获取HF Pro账户（$9/月）
2. 设置HF_TOKEN环境变量  
3. 复制命令运行即可！

**就是这么简单！** 🎉