# LM Studio 使用指南

## 🎉 转换完成！

你的模型已经成功合并并准备好使用了！

## 📁 可用模型

### 1. 合并后的模型 (推荐)
- **路径**: `./gemma3-1b-tool-use-merged/`
- **格式**: Transformers格式 (safetensors)
- **大小**: ~3.7GB
- **兼容性**: 完全兼容所有推理框架

### 2. 临时模型
- **路径**: `./temp_model_for_gguf/`
- **格式**: Transformers格式 (safetensors)
- **大小**: ~3.7GB
- **用途**: 用于进一步转换

## 🚀 在LM Studio中使用

### 步骤1: 打开LM Studio
1. 下载并安装 [LM Studio](https://lmstudio.ai/)
2. 启动LM Studio

### 步骤2: 加载模型
1. 点击 "Local Server" 标签
2. 点击 "Browse" 按钮
3. 选择模型文件夹: `./gemma3-1b-tool-use-merged/`
4. 点击 "Load Model"

### 步骤3: 配置参数
在 "Chat" 标签中设置：
- **Temperature**: 0.7-0.9
- **Top P**: 0.9
- **Max Tokens**: 512
- **Stop Sequences**: `</s>`, `<eos>`
- **Context Length**: 4096

### 步骤4: 开始对话
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

## 🔧 其他使用方法

### 方法1: Python代码
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载模型
model = AutoModelForCausalLM.from_pretrained("./gemma3-1b-tool-use-merged")
tokenizer = AutoTokenizer.from_pretrained("./gemma3-1b-tool-use-merged")

# 进行推理
prompt = "帮我查询北京的天气"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### 方法2: 使用uv运行
```bash
# 创建推理脚本
echo '
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./gemma3-1b-tool-use-merged")
tokenizer = AutoTokenizer.from_pretrained("./gemma3-1b-tool-use-merged")

prompt = "帮我查询北京的天气"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
' > inference.py

# 运行推理
uv run python inference.py
```

## 🎯 工具调用格式

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

## 💡 提示

1. **首次加载**: 模型首次加载可能需要几分钟
2. **内存要求**: 建议至少8GB RAM
3. **GPU加速**: 如果有GPU，LM Studio会自动使用
4. **模型大小**: 3.7GB，确保有足够存储空间

## 🎉 恭喜！

你的Gemma-3-1b工具调用模型已经准备就绪！现在可以在LM Studio中享受工具调用功能了！
