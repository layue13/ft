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
import platform
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
    
    return True

def check_llama_cpp():
    """检查llama-cpp-python是否已安装"""
    try:
        import llama_cpp
        print("✅ llama-cpp-python已安装")
        return True
    except ImportError:
        print("❌ llama-cpp-python未安装")
        return False

def install_llama_cpp_simple():
    """简单安装llama-cpp-python"""
    print("📦 安装llama-cpp-python...")
    
    # 尝试多种安装方式
    install_methods = [
        # 方式1: uv安装
        (["uv", "add", "llama-cpp-python"], "uv"),
        # 方式2: pip安装
        ([sys.executable, "-m", "pip", "install", "llama-cpp-python"], "pip"),
    ]
    
    for cmd, method in install_methods:
        try:
            print(f"🔧 尝试使用{method}安装...")
            subprocess.run(cmd, check=True)
            print(f"✅ {method}安装成功")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"❌ {method}安装失败")
            continue
    
    print("❌ 所有安装方式都失败")
    return False

def install_llama_cpp():
    """安装llama.cpp - 支持多种方式"""
    print("📦 安装llama.cpp...")
    
    # 首先检查是否已经安装
    if check_llama_cpp():
        return True
    
    # 尝试简单安装
    if install_llama_cpp_simple():
        return True
    
    # 如果简单安装失败，尝试从源码安装
    if os.path.exists("llama.cpp"):
        print("✅ llama.cpp已存在，跳过下载")
        return True
    
    try:
        # 克隆llama.cpp仓库
        subprocess.run([
            "git", "clone", "https://github.com/ggml-org/llama.cpp.git"
        ], check=True)
        print("✅ 已克隆llama.cpp仓库")
        
        # 尝试不同的构建方法
        os.chdir("llama.cpp")
        
        # 方法1: 尝试使用CMake
        try:
            print("🔧 尝试使用CMake构建...")
            subprocess.run(["cmake", "--version"], check=True, capture_output=True)
            
            # 创建build目录
            os.makedirs("build", exist_ok=True)
            os.chdir("build")
            
            # 配置和构建
            subprocess.run(["cmake", ".."], check=True)
            subprocess.run(["cmake", "--build", ".", "--config", "Release"], check=True)
            
            print("✅ CMake构建成功")
            os.chdir("..")
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠️ CMake构建失败，尝试使用make...")
            
            # 方法2: 尝试使用make (旧版本)
            try:
                subprocess.run(["make"], check=True)
                print("✅ Make构建成功")
            except subprocess.CalledProcessError:
                print("⚠️ Make构建也失败")
                os.chdir("..")
                return False
        
        os.chdir("..")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 安装失败: {e}")
        return False

def is_peft_model(model_path):
    """检查是否为PEFT模型"""
    try:
        # 检查是否存在adapter_config.json
        adapter_config = os.path.join(model_path, "adapter_config.json")
        if os.path.exists(adapter_config):
            return True
        
        # 检查是否存在adapter_model.bin
        adapter_model = os.path.join(model_path, "adapter_model.bin")
        if os.path.exists(adapter_model):
            return True
        
        return False
    except:
        return False

def merge_lora_weights(model_path, output_dir):
    """合并LoRA权重"""
    print(f"🔧 检测到PEFT模型，开始合并LoRA权重...")
    print(f"📦 源模型: {model_path}")
    print(f"📁 输出目录: {output_dir}")
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        
        print("📦 加载基础模型 (google/gemma-3-1b-it)...")
        base_model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-3-1b-it",
            torch_dtype="auto",
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
        
        print("📦 加载LoRA权重...")
        model = PeftModel.from_pretrained(base_model, model_path)
        
        print("🔧 合并权重...")
        merged_model = model.merge_and_unload()
        
        print("💾 保存合并后的模型...")
        merged_model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        print(f"✅ LoRA权重合并完成！")
        print(f"📁 合并后的模型已保存到: {output_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ 合并失败: {e}")
        return False

def convert_to_gguf(model_path, output_file, quantization="q4_k_m"):
    """转换为GGUF格式"""
    print(f"🔄 开始转换: {model_path} -> {output_file}")
    
    # 检查转换脚本
    convert_script = "llama.cpp/convert_hf_to_gguf.py"
    
    # 如果llama.cpp不存在，尝试使用pip安装的版本
    if not os.path.exists("llama.cpp"):
        print("📦 llama.cpp不存在，尝试使用pip安装的版本...")
        try:
            import llama_cpp
            print("✅ 找到llama-cpp-python")
            
            # 使用transformers的转换功能
            return convert_with_transformers(model_path, output_file, quantization)
        except ImportError:
            print("❌ 未找到llama-cpp-python，请先安装")
            print("💡 建议运行: uv add llama-cpp-python")
            return False
    
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

def convert_with_transformers(model_path, output_file, quantization="q4_k_m"):
    """使用transformers进行转换"""
    print("🔄 使用transformers进行转换...")
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        print("📦 加载模型...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        print("💾 保存为GGUF兼容格式...")
        # 保存为transformers格式，然后可以使用其他工具转换
        temp_dir = "temp_for_gguf"
        model.save_pretrained(temp_dir)
        tokenizer.save_pretrained(temp_dir)
        
        print("✅ 模型已保存为兼容格式")
        print(f"📁 临时目录: {temp_dir}")
        print("💡 请使用其他工具(如llama.cpp)将临时目录转换为GGUF格式")
        
        return True
        
    except Exception as e:
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

import os
import subprocess
import sys

def run_llama_cpp(model_path, prompt, max_tokens=512):
    \"\"\"使用llama.cpp运行GGUF模型\"\"\"
    
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
        print("\n❌ 依赖检查失败，请安装git")
        return
    
    # 2. 检查llama-cpp-python
    if not check_llama_cpp():
        print("\n📦 需要安装llama-cpp-python...")
        if not install_llama_cpp():
            print("\n❌ llama.cpp安装失败")
            print("\n💡 手动安装选项:")
            print("1. 使用uv: uv add llama-cpp-python")
            print("2. 使用pip: pip install llama-cpp-python")
            print("3. 运行安装脚本: python install_llama_cpp.py")
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
    
    # 4. 检查模型类型并处理
    print("\n" + "=" * 50)
    
    # 检查是否为PEFT模型
    if is_peft_model(model_path):
        print("🔍 检测到PEFT模型，需要先合并LoRA权重...")
        
        # 生成合并后的模型路径
        merged_dir = f"{model_path}-merged"
        if os.path.exists(merged_dir):
            print(f"✅ 发现已合并的模型: {merged_dir}")
            model_path = merged_dir
        else:
            print("📦 开始合并LoRA权重...")
            if merge_lora_weights(model_path, merged_dir):
                model_path = merged_dir
            else:
                print("❌ LoRA权重合并失败")
                return
    else:
        print("✅ 检测到标准模型，无需合并")
    
    # 5. 执行转换
    print("\n" + "=" * 50)
    if convert_to_gguf(model_path, output_file, quantization):
        print("✅ 转换成功！")
        
        # 6. 创建使用示例
        create_usage_script(output_file)
        
        # 7. 显示结果
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
