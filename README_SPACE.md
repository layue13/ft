# Hugging Face Spaces 部署指南

## 🚀 部署到 Hugging Face Spaces

### 步骤 1: 准备项目

1. **确保项目完整**
   ```bash
   # 检查项目结构
   ls -la
   # 应该包含: src/, configs/, scripts/, app.py, pyproject.toml
   ```

2. **初始化Git仓库** (如果还没有)
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Gemma3-1b tool calling finetuning"
   ```

### 步骤 2: 推送到 Hugging Face Hub

1. **创建Hugging Face仓库**
   - 访问 https://huggingface.co/new
   - 选择 "Repository" 类型
   - 设置仓库名称，如 `gemma3-tool-finetuning`
   - 选择 "Public" 或 "Private"

2. **推送代码**
   ```bash
   # 添加远程仓库
   git remote add origin https://huggingface.co/YOUR_USERNAME/gemma3-tool-finetuning
   
   # 推送代码
   git push -u origin main
   ```

### 步骤 3: 创建 Space

1. **创建新的Space**
   - 访问 https://huggingface.co/spaces
   - 点击 "Create new Space"
   - 选择 "Gradio" SDK
   - 选择 "Python" 运行时
   - 设置Space名称，如 `gemma3-tool-finetuning-space`

2. **配置Space设置**
   - **Hardware**: 选择 "GPU" (T4 Small 或更高)
   - **Python packages**: 会自动从 `pyproject.toml` 安装
   - **Environment variables**: 添加 `HF_TOKEN` (如果需要)

### 步骤 4: 申请GPU资助 (可选)

如果您需要免费GPU：

1. **进入Space设置**
   - 在Space页面点击 "Settings"
   - 找到 "Sleep time" 设置
   - 点击 "Ask for community grant"

2. **填写申请表单**
   - 描述您的项目用途
   - 说明为什么需要GPU
   - 提交申请

### 步骤 5: 使用Space

1. **启动Space**
   - Space会自动构建和启动
   - 等待构建完成 (通常需要几分钟)

2. **开始训练**
   - 在Web界面中点击 "🚀 开始训练"
   - 监控训练进度
   - 查看训练日志

## 🔧 Space配置文件

### requirements.txt (自动生成)
Space会自动从 `pyproject.toml` 生成 `requirements.txt`，包含所有依赖。

### app.py
已经配置好的Gradio应用，包含：
- 训练配置界面
- 实时训练日志
- 进度监控

## 📊 监控和调试

### 查看日志
- 在Space页面点击 "Logs" 标签
- 实时查看训练进度
- 调试任何错误

### 查看文件
- 在Space页面点击 "Files" 标签
- 查看生成的模型文件
- 下载训练结果

## 💰 成本控制

### 免费选项
- **社区GPU资助**: 申请免费GPU使用时间
- **CPU训练**: 使用CPU进行小规模测试

### 付费选项
- **T4 Small**: $0.40/小时
- **A10G Small**: $1.00/小时
- **A10G Large**: $1.50/小时

## 🚨 注意事项

1. **模型访问权限**
   - 确保您的HF账户有Gemma3-1b-it的访问权限
   - 在Space设置中添加 `HF_TOKEN`

2. **存储限制**
   - Space有存储限制，注意模型大小
   - 考虑使用LoRA等参数高效方法

3. **时间限制**
   - 免费Space有使用时间限制
   - 付费Space可以持续运行

4. **数据隐私**
   - 确保数据集可以公开访问
   - 或使用私有数据集配置

## 🔄 更新和维护

### 更新代码
```bash
# 修改代码后
git add .
git commit -m "Update: improve training process"
git push origin main
```

### 重启Space
- 在Space设置中点击 "Restart"
- 或等待自动重启

## 📞 获取帮助

- **Hugging Face文档**: https://huggingface.co/docs
- **Space文档**: https://huggingface.co/docs/hub/spaces
- **社区论坛**: https://discuss.huggingface.co

## 🎯 下一步

1. 按照上述步骤部署Space
2. 申请GPU资助 (如果需要)
3. 开始训练您的模型
4. 监控训练进度
5. 下载和测试微调后的模型

祝您训练顺利！🚀
