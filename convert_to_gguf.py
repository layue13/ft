#!/usr/bin/env python3
"""
GGUF格式转换脚本
使用llama.cpp的convert_hf_to_gguf.py将Hugging Face模型转换为GGUF格式
支持PEFT模型和多种量化选项
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def check_dependencies():
    """检查必要的依赖"""
    print("🔍 检查依赖...")
    
    # 检查git
    try:
        subprocess.run(["git", "--version"], check=True, capture_output=True)
        print("✅ git")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ git 未安装")
        return False
    
    # 检查make
    try:
        subprocess.run(["make", "--version"], check=True, capture_output=True)
        print("✅ make")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ make 未安装")
        return False
    
    return True

def install_llama_cpp():
    """安装llama.cpp"""
    print("📦 安装llama.cpp...")
    
    if os.path.exists("llama.cpp"):
        print("✅ llama.cpp已存在，跳过下载")
        return True
    
    try:
        # 克隆llama.cpp仓库
        subprocess.run([
            "git", "clone", "https://github.com/ggml-org/llama.cpp.git"
        ], check=True)
        print("✅ 已克隆llama.cpp仓库")
        
        # 编译
        os.chdir("llama.cpp")
        subprocess.run(["make"], check=True)
        print("✅ 已编译llama.cpp")
        os.chdir("..")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 安装失败: {e}")
        return False

def convert_to_gguf(model_path, output_file, quantization="q4_k_m"):
    """转换为GGUF格式"""
    print(f"🔄 开始转换: {model_path} -> {output_file}")
    
    if not os.path.exists("llama.cpp"):
        print("❌ llama.cpp未安装，请先运行安装")
        return False
    
    # 检查转换脚本
    convert_script = "llama.cpp/convert_hf_to_gguf.py"
    if not os.path.exists(convert_script):
        print(f"❌ 转换脚本不存在: {convert_script}")
        return False
    
    try:
        # 运行转换
        cmd = [
            sys.executable, convert_script,
            model_path,
            "--outfile", output_file,
            "--outtype", quantization
        ]
        
        print(f"🚀 执行命令: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
        print(f"✅ 转换完成: {output_file}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 转换失败: {e}")
        return False

def create_usage_script(output_file):
    """创建使用示例脚本"""
    print("📝 创建使用示例脚本...")
    
    usage_script = f"""
#!/usr/bin/env python3
\"\"\"
GGUF模型使用示例
\"\"\"

import subprocess
import sys

def run_llama_cpp(model_path, prompt, max_tokens=512):
    \"\"\"使用llama.cpp运行GGUF模型\"\"\"
    cmd = [
        "./llama.cpp/main",
        "-m", model_path,
        "-n", str(max_tokens),
        "-p", prompt,
        "--repeat_penalty", "1.1",
        "--temp", "0.7"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"运行失败: {{e}}")
        return None

def main():
    model_path = "{output_file}"
    
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {{model_path}}")
        return
    
    # 示例提示
    prompt = "帮我查询北京的天气"
    
    print(f"用户: {{prompt}}")
    print("助手: ", end="")
    
    response = run_llama_cpp(model_path, prompt)
    if response:
        print(response)
    else:
        print("生成失败")

if __name__ == "__main__":
    main()
"""
    
    with open("run_gguf_model.py", "w") as f:
        f.write(usage_script)
    
    print("✅ 使用示例脚本已创建: run_gguf_model.py")

def main():
    print("🚀 GGUF格式转换工具")
    print("=" * 50)
    
    # 1. 检查依赖
    if not check_dependencies():
        print("\n❌ 依赖检查失败，请安装必要的工具")
        return
    
    # 2. 安装llama.cpp
    if not install_llama_cpp():
        print("\n❌ llama.cpp安装失败")
        return
    
    # 3. 获取输入
    print("\n📝 配置转换参数:")
    
    model_path = input("请输入模型路径 (例如: ./gemma3-1b-tool-use-merged 或 layue13/gemma3-1b-tool-use): ").strip()
    if not model_path:
        model_path = "./gemma3-1b-tool-use-merged"
    
    output_file = input("请输入输出文件名 (例如: gemma3-1b-tool-use.gguf): ").strip()
    if not output_file:
        output_file = "gemma3-1b-tool-use.gguf"
    
    # 量化选项
    quantization_options = {
        "1": "q4_k_m",    # 推荐，平衡大小和性能
        "2": "q8_0",      # 高质量，较大文件
        "3": "q5_k_m",    # 中等质量
        "4": "q3_k_m",    # 小文件，较低质量
    }
    
    print("\n🔧 选择量化类型:")
    for key, value in quantization_options.items():
        print(f"  {key}. {value}")
    
    choice = input("请选择 (默认1): ").strip() or "1"
    quantization = quantization_options.get(choice, "q4_k_m")
    
    print(f"\n📊 转换配置:")
    print(f"  模型路径: {model_path}")
    print(f"  输出文件: {output_file}")
    print(f"  量化类型: {quantization}")
    
    # 4. 执行转换
    print("\n" + "=" * 50)
    if convert_to_gguf(model_path, output_file, quantization):
        print("✅ 转换成功！")
        
        # 5. 创建使用示例
        create_usage_script(output_file)
        
        # 6. 显示结果
        print("\n" + "=" * 50)
        print("🎉 转换完成！")
        print(f"\n📁 输出文件: {output_file}")
        print(f"📝 使用示例: run_gguf_model.py")
        
        print("\n🔧 使用方法:")
        print("1. 直接使用llama.cpp:")
        print(f"   ./llama.cpp/main -m {output_file} -n 512 -p '你的提示'")
        print("\n2. 使用示例脚本:")
        print("   python run_gguf_model.py")
        print("\n3. 在LM Studio中加载:")
        print(f"   选择文件: {output_file}")
        
    else:
        print("❌ 转换失败")

if __name__ == "__main__":
    main()
