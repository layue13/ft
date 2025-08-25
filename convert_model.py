#!/usr/bin/env python3
"""
模型转换脚本 - 支持LoRA权重合并和GGUF转换
基于第一性原理：兼容性 + 性能 + 易用性
"""

import os
import subprocess
import sys
from pathlib import Path

def install_dependencies():
    """安装必要的依赖"""
    print("📦 安装转换依赖...")
    
    dependencies = [
        "transformers",
        "peft", 
        "torch",
        "transformers-to-gguf"
    ]
    
    for dep in dependencies:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            print(f"✅ 已安装 {dep}")
        except subprocess.CalledProcessError:
            print(f"⚠️ 安装 {dep} 失败")

def merge_lora_weights(model_name, output_dir):
    """合并LoRA权重"""
    print(f"🔧 合并LoRA权重: {model_name}")
    
    merge_script = f"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

print("📦 加载基础模型...")
base_model = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-it")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")

print("📦 加载LoRA权重...")
model = PeftModel.from_pretrained(base_model, "{model_name}")

print("🔧 合并权重...")
merged_model = model.merge_and_unload()

print("💾 保存合并后的模型...")
merged_model.save_pretrained("{output_dir}")
tokenizer.save_pretrained("{output_dir}")

print("✅ LoRA权重合并完成！")
"""
    
    with open("temp_merge.py", "w") as f:
        f.write(merge_script)
    
    try:
        subprocess.check_call([sys.executable, "temp_merge.py"])
        print(f"✅ 模型已保存到: {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"❌ 合并失败: {e}")
        return False
    finally:
        if os.path.exists("temp_merge.py"):
            os.remove("temp_merge.py")
    
    return True

def convert_to_gguf(model_path, output_file):
    """转换为GGUF格式"""
    print(f"🔄 转换为GGUF: {model_path} -> {output_file}")
    
    try:
        # 使用transformers-to-gguf
        cmd = [
            "transformers-to-gguf",
            model_path,
            "--output", output_file,
            "--quantize", "q4_k_m"
        ]
        
        subprocess.check_call(cmd)
        print(f"✅ GGUF转换完成: {output_file}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ GGUF转换失败: {e}")
        return False

def create_mlx_script(model_path):
    """创建MLX使用脚本"""
    print("📝 创建MLX使用脚本...")
    
    mlx_script = f"""
#!/usr/bin/env python3
\"\"\"
MLX推理脚本 - 用于Apple Silicon优化
\"\"\"

import mlx.core as mx
from transformers import AutoTokenizer
import json

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained("{model_path}")

def generate_tool_call(prompt, max_length=512):
    \"\"\"生成工具调用响应\"\"\"
    # 格式化输入
    formatted_prompt = f"<bos><start_of_turn>user\\n{{prompt}}<end_of_turn>\\n<start_of_turn>model\\n"
    
    # 这里需要MLX模型加载和推理
    # 注意: 需要先将模型转换为MLX格式
    print("⚠️ 需要先将模型转换为MLX格式")
    print("📝 使用转换后的模型进行推理")
    
    return "工具调用响应"

# 示例使用
if __name__ == "__main__":
    prompt = "帮我查询北京的天气"
    response = generate_tool_call(prompt)
    print(f"用户: {{prompt}}")
    print(f"助手: {{response}}")
"""
    
    with open("mlx_inference.py", "w") as f:
        f.write(mlx_script)
    
    print("✅ MLX脚本已创建: mlx_inference.py")

def main():
    print("🚀 开始模型转换...")
    
    # 配置
    model_name = input("请输入模型名称 (例如: layue13/gemma3-1b-tool-use): ").strip()
    if not model_name:
        model_name = "layue13/gemma3-1b-tool-use"
    
    output_dir = "./gemma3-1b-tool-use-merged"
    gguf_file = "./gemma3-1b-tool-use.gguf"
    
    # 1. 安装依赖
    install_dependencies()
    
    # 2. 合并LoRA权重
    print("\n" + "="*50)
    if merge_lora_weights(model_name, output_dir):
        print("✅ LoRA权重合并成功")
    else:
        print("❌ LoRA权重合并失败")
        return
    
    # 3. 转换为GGUF
    print("\n" + "="*50)
    if convert_to_gguf(output_dir, gguf_file):
        print("✅ GGUF转换成功")
    else:
        print("❌ GGUF转换失败")
    
    # 4. 创建MLX脚本
    print("\n" + "="*50)
    create_mlx_script(output_dir)
    
    # 5. 使用说明
    print("\n" + "="*50)
    print("🎉 转换完成！")
    print("\n📁 输出文件:")
    print(f"  - 合并模型: {output_dir}")
    print(f"  - GGUF文件: {gguf_file}")
    print(f"  - MLX脚本: mlx_inference.py")
    
    print("\n🔧 使用方法:")
    print("1. 合并模型: 直接使用transformers加载")
    print("2. GGUF文件: 使用llama.cpp或LM Studio加载")
    print("3. MLX脚本: 在Apple Silicon Mac上运行")
    
    print("\n💡 推荐:")
    print("- 本地推理: 使用GGUF格式")
    print("- Apple Silicon: 使用MLX优化")
    print("- 生产部署: 使用合并后的模型")

if __name__ == "__main__":
    main()
