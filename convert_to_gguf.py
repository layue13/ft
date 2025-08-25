#!/usr/bin/env python3
"""
GGUF格式转换脚本
将合并后的模型转换为GGUF格式，获得最佳性能和兼容性
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """检查必要的依赖"""
    print("🔍 检查依赖...")
    
    try:
        import transformers
        print("✅ transformers")
    except ImportError:
        print("❌ transformers 未安装")
        return False
    
    try:
        # 检查transformers-to-gguf
        result = subprocess.run(["transformers-to-gguf", "--help"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ transformers-to-gguf")
        else:
            print("❌ transformers-to-gguf 未安装")
            return False
    except FileNotFoundError:
        print("❌ transformers-to-gguf 未安装")
        return False
    
    return True

def install_dependencies():
    """安装必要的依赖"""
    print("📦 安装依赖...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers-to-gguf"])
        print("✅ transformers-to-gguf 安装成功")
        return True
    except subprocess.CalledProcessError:
        print("❌ 安装失败")
        return False

def convert_to_gguf(model_path, output_file, quantization="q4_k_m"):
    """转换为GGUF格式"""
    print(f"🔄 开始GGUF转换...")
    print(f"📦 源模型: {model_path}")
    print(f"📁 输出文件: {output_file}")
    print(f"🔧 量化类型: {quantization}")
    
    # 检查源模型是否存在
    if not os.path.exists(model_path):
        print(f"❌ 源模型不存在: {model_path}")
        return False
    
    # 检查是否包含必要的文件
    required_files = ["config.json", "pytorch_model.bin", "tokenizer.json"]
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_path, f))]
    
    if missing_files:
        print(f"❌ 源模型缺少必要文件: {missing_files}")
        print("💡 请先运行 merge_lora.py 合并LoRA权重")
        return False
    
    try:
        # 使用transformers-to-gguf转换
        cmd = [
            "transformers-to-gguf",
            model_path,
            "--output", output_file,
            "--quantize", quantization
        ]
        
        print("🔄 执行转换命令...")
        print(f"命令: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ GGUF转换成功！")
            
            # 显示文件信息
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
                print(f"📁 输出文件: {output_file}")
                print(f"📊 文件大小: {file_size:.2f} MB")
            
            return True
        else:
            print("❌ GGUF转换失败")
            print(f"错误输出: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        return False

def create_usage_script(output_file):
    """创建使用脚本"""
    print("📝 创建使用脚本...")
    
    script_content = f"""#!/usr/bin/env python3
\"\"\"
GGUF模型使用示例
\"\"\"

import os
from llama_cpp import Llama

def load_gguf_model(model_path):
    \"\"\"加载GGUF模型\"\"\"
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {{model_path}}")
        return None
    
    try:
        # 加载模型
        llm = Llama(
            model_path=model_path,
            n_ctx=4096,  # 上下文长度
            n_threads=4,  # CPU线程数
            n_gpu_layers=0  # GPU层数，根据你的GPU调整
        )
        print("✅ GGUF模型加载成功")
        return llm
    except Exception as e:
        print(f"❌ 模型加载失败: {{e}}")
        return None

def generate_tool_call(llm, prompt, max_tokens=512):
    \"\"\"生成工具调用响应\"\"\"
    try:
        # 格式化输入
        formatted_prompt = f"<bos><start_of_turn>user\\n{{prompt}}<end_of_turn>\\n<start_of_turn>model\\n"
        
        # 生成响应
        response = llm(
            formatted_prompt,
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            stop=["<eos>", "</s>"]
        )
        
        return response['choices'][0]['text']
    except Exception as e:
        print(f"❌ 生成失败: {{e}}")
        return None

if __name__ == "__main__":
    # 加载模型
    model_path = "{output_file}"
    llm = load_gguf_model(model_path)
    
    if llm:
        # 测试工具调用
        prompt = "帮我查询北京的天气"
        print(f"\\n用户: {{prompt}}")
        
        response = generate_tool_call(llm, prompt)
        if response:
            print(f"助手: {{response}}")
        else:
            print("❌ 生成失败")
"""
    
    script_file = "gguf_inference.py"
    with open(script_file, "w") as f:
        f.write(script_content)
    
    print(f"✅ 使用脚本已创建: {script_file}")

def main():
    print("🚀 GGUF格式转换工具")
    print("=" * 50)
    
    # 检查依赖
    if not check_dependencies():
        print("\n📦 安装缺失的依赖...")
        if not install_dependencies():
            print("❌ 依赖安装失败")
            return
        if not check_dependencies():
            print("❌ 依赖检查失败")
            return
    
    print("\n" + "=" * 50)
    
    # 获取输入
    model_path = input("请输入模型路径 (例如: ./gemma3-1b-tool-use-merged): ").strip()
    if not model_path:
        print("❌ 模型路径不能为空")
        return
    
    output_file = input("请输入输出文件 (默认: ./gemma3-1b-tool-use.gguf): ").strip()
    if not output_file:
        output_file = "./gemma3-1b-tool-use.gguf"
    
    # 量化选项
    print("\n🔧 选择量化类型:")
    print("1. q4_k_m (推荐) - 平衡质量和大小")
    print("2. q8_0 - 高质量，较大文件")
    print("3. q5_k_m - 中等质量")
    print("4. q3_k_m - 小文件，质量较低")
    
    quantization = input("请选择量化类型 (默认: q4_k_m): ").strip()
    if not quantization:
        quantization = "q4_k_m"
    
    print("\n" + "=" * 50)
    
    # 执行转换
    if convert_to_gguf(model_path, output_file, quantization):
        print("\n🎉 GGUF转换成功！")
        print(f"📁 GGUF文件: {output_file}")
        
        # 创建使用脚本
        create_usage_script(output_file)
        
        print("\n💡 使用方法:")
        print("1. 安装依赖: pip install llama-cpp-python")
        print("2. 运行示例: python gguf_inference.py")
        print("3. 在LM Studio中加载GGUF文件")
        print("4. 使用llama.cpp进行推理")
    else:
        print("\n❌ GGUF转换失败，请检查错误信息")

if __name__ == "__main__":
    main()
