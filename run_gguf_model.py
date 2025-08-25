
#!/usr/bin/env python3
"""
GGUF模型使用示例
"""

import os
import subprocess
import sys

def run_llama_cpp(model_path, prompt, max_tokens=512):
    """使用llama.cpp运行GGUF模型"""
    
    # 尝试多种运行方式
    commands = [
        # 方式1: 本地llama.cpp
        ["./llama.cpp/main", "-m", model_path, "-n", str(max_tokens), "-p", prompt],
        # 方式2: 系统安装的llama.cpp
        ["llama-cpp", "-m", model_path, "-n", str(max_tokens), "-p", prompt],
        # 方式3: python包
        [sys.executable, "-m", "llama_cpp", "-m", model_path, "-n", str(max_tokens), "-p", prompt]
    ]
    
    for cmd in commands:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    
    print("❌ 所有运行方式都失败")
    return None

def main():
    model_path = "gemma3-1b-tool-use"
    
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        return
    
    # 示例提示
    prompt = "帮我查询北京的天气"
    
    print(f"用户: {prompt}")
    print("助手: ", end="")
    
    response = run_llama_cpp(model_path, prompt)
    if response:
        print(response)
    else:
        print("生成失败")

if __name__ == "__main__":
    main()
