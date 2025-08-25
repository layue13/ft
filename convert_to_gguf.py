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
        # 检查llama-cpp-python (用于GGUF推理)
        import llama_cpp
        print("✅ llama-cpp-python")
    except ImportError:
        print("❌ llama-cpp-python 未安装")
        return False
    
    try:
        # 检查ctransformers (用于GGUF转换)
        import ctransformers
        print("✅ ctransformers")
    except ImportError:
        print("❌ ctransformers 未安装")
        return False
    
    return True

def install_dependencies():
    """安装必要的依赖"""
    print("📦 安装依赖...")
    
    dependencies = [
        "llama-cpp-python",
        "ctransformers[cuda]"  # 支持CUDA加速
    ]
    
    for dep in dependencies:
        try:
            # 使用uv pip安装
            subprocess.check_call(["uv", "pip", "install", dep])
            print(f"✅ {dep} 安装成功")
        except subprocess.CalledProcessError:
            print(f"❌ {dep} 安装失败")
            return False
    
    return True

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
    required_files = ["config.json", "tokenizer.json"]
    # 检查模型文件（支持多种格式）
    model_files = ["pytorch_model.bin", "model.safetensors"]
    has_model_file = any(os.path.exists(os.path.join(model_path, f)) for f in model_files)
    
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_path, f))]
    
    if missing_files:
        print(f"❌ 源模型缺少必要文件: {missing_files}")
        print("💡 请先运行 merge_lora.py 合并LoRA权重")
        return False
    
    if not has_model_file:
        print(f"❌ 源模型缺少模型文件，需要以下之一: {model_files}")
        print("💡 请先运行 merge_lora.py 合并LoRA权重")
        return False
    
    print("✅ 模型文件检查通过")
    
    try:
        # 使用transformers加载，然后转换为GGUF
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print("🔄 使用transformers加载模型...")
        
        # 加载模型和tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        print("🔄 转换为GGUF格式...")
        
        # 使用llama.cpp的转换工具
        try:
            # 尝试使用llama.cpp的convert.py
            convert_script = """
import sys
import os
sys.path.append('llama.cpp')

from convert import convert_hf_to_gguf

# 转换模型
convert_hf_to_gguf(
    model_path='{model_path}',
    output_path='{output_file}',
    model_type='llama'
)
"""
            
            # 检查是否有llama.cpp
            if not os.path.exists("llama.cpp"):
                print("📦 下载llama.cpp...")
                subprocess.check_call(["git", "clone", "https://github.com/ggerganov/llama.cpp.git"])
            
            # 运行转换
            with open("temp_convert.py", "w") as f:
                f.write(convert_script.format(model_path=model_path, output_file=output_file))
            
            subprocess.check_call([sys.executable, "temp_convert.py"])
            os.remove("temp_convert.py")
            
        except Exception as e:
            print(f"llama.cpp转换失败: {e}")
            print("🔄 尝试使用transformers直接保存...")
            
            # 备用方案：直接保存为transformers格式，然后手动转换
            temp_dir = "./temp_model_for_gguf"
            model.save_pretrained(temp_dir)
            tokenizer.save_pretrained(temp_dir)
            
            print(f"💾 模型已保存到临时目录: {temp_dir}")
            print("💡 请手动使用llama.cpp转换:")
            print(f"   cd llama.cpp")
            print(f"   python convert.py {temp_dir} --outfile {output_file} --outtype {quantization}")
            return False
        
        print("✅ GGUF转换成功！")
        
        # 显示文件信息
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
            print(f"📁 输出文件: {output_file}")
            print(f"📊 文件大小: {file_size:.2f} MB")
        
        return True
            
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        print("💡 提示: 如果转换失败，可以尝试使用llama.cpp手动转换")
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
    print("5. 无量化 - 保持原始精度")
    
    quantization = input("请选择量化类型 (默认: q4_k_m): ").strip()
    if not quantization:
        quantization = "q4_k_m"
    
    # 注意：ctransformers的量化选项可能不同
    print(f"💡 注意: 使用ctransformers进行转换，量化选项: {quantization}")
    
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
