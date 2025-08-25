#!/usr/bin/env python3
"""
手动转换脚本 - 将合并后的模型转换为GGUF格式
"""

import os
import subprocess
import sys

def main():
    print("🚀 开始手动转换...")
    
    model_path = "./gemma3-1b-tool-use-merged"
    output_file = "./gemma3-1b-tool-use.gguf"
    
    # 检查模型是否存在
    if not os.path.exists(model_path):
        print(f"❌ 模型不存在: {model_path}")
        return
    
    print(f"📦 源模型: {model_path}")
    print(f"📁 输出文件: {output_file}")
    
    # 方法1: 尝试使用llama.cpp的转换脚本
    print("\n🔄 方法1: 使用llama.cpp转换...")
    try:
        # 使用系统Python运行llama.cpp转换
        cmd = [
            sys.executable, 
            "llama.cpp/convert_hf_to_gguf.py",
            model_path,
            "--outfile", output_file,
            "--outtype", "q4_k_m"
        ]
        
        print(f"执行命令: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ 转换成功！")
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
                print(f"📁 输出文件: {output_file}")
                print(f"📊 文件大小: {file_size:.2f} MB")
            return
        else:
            print(f"❌ 转换失败: {result.stderr}")
    except Exception as e:
        print(f"❌ 方法1失败: {e}")
    
    # 方法2: 使用transformers保存为临时格式
    print("\n🔄 方法2: 使用transformers保存...")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print("📦 加载模型...")
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        temp_dir = "./temp_model_for_gguf"
        print(f"💾 保存到临时目录: {temp_dir}")
        
        model.save_pretrained(temp_dir)
        tokenizer.save_pretrained(temp_dir)
        
        print("✅ 模型已保存到临时目录")
        print(f"📁 临时目录: {temp_dir}")
        
        # 提供手动转换指导
        print("\n💡 手动转换步骤:")
        print("1. 安装llama.cpp依赖:")
        print("   pip install torch transformers sentencepiece")
        print("2. 运行转换:")
        print(f"   python llama.cpp/convert_hf_to_gguf.py {temp_dir} --outfile {output_file} --outtype q4_k_m")
        
    except Exception as e:
        print(f"❌ 方法2失败: {e}")
    
    # 方法3: 提供其他转换选项
    print("\n🔄 方法3: 其他转换选项...")
    print("💡 可以尝试以下方法:")
    print("1. 使用LM Studio直接加载合并后的模型")
    print("2. 使用Ollama转换")
    print("3. 使用其他GGUF转换工具")

if __name__ == "__main__":
    main()
