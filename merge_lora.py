#!/usr/bin/env python3
"""
LoRA权重合并脚本
将LoRA权重合并到基础模型中，获得完全兼容的模型
"""

import os
import sys
from pathlib import Path

def merge_lora_weights(model_name, output_dir):
    """合并LoRA权重到基础模型"""
    print(f"🔧 开始合并LoRA权重...")
    print(f"📦 源模型: {model_name}")
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
        model = PeftModel.from_pretrained(base_model, model_name)
        
        print("🔧 合并权重...")
        merged_model = model.merge_and_unload()
        
        print("💾 保存合并后的模型...")
        # 确保输出目录存在
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        merged_model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        print(f"✅ LoRA权重合并完成！")
        print(f"📁 模型已保存到: {output_dir}")
        
        # 显示模型信息
        total_params = sum(p.numel() for p in merged_model.parameters())
        print(f"📊 模型参数总数: {total_params:,}")
        
        return True
        
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("请运行: pip install transformers peft torch")
        return False
    except Exception as e:
        print(f"❌ 合并失败: {e}")
        return False

def main():
    print("🚀 LoRA权重合并工具")
    print("=" * 50)
    
    # 获取输入
    model_name = input("请输入模型名称 (例如: layue13/gemma3-1b-tool-use): ").strip()
    if not model_name:
        print("❌ 模型名称不能为空")
        return
    
    output_dir = input("请输入输出目录 (默认: ./gemma3-1b-tool-use-merged): ").strip()
    if not output_dir:
        output_dir = "./gemma3-1b-tool-use-merged"
    
    print("\n" + "=" * 50)
    
    # 执行合并
    if merge_lora_weights(model_name, output_dir):
        print("\n🎉 合并成功！")
        print(f"📁 合并后的模型: {output_dir}")
        print("\n💡 下一步:")
        print("1. 使用合并后的模型进行推理")
        print("2. 运行 convert_to_gguf.py 转换为GGUF格式")
        print("3. 在LM Studio中加载使用")
    else:
        print("\n❌ 合并失败，请检查错误信息")

if __name__ == "__main__":
    main()
