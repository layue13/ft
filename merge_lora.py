#!/usr/bin/env python3
"""
LoRA权重合并脚本
将PEFT模型合并为完整模型，解决LM Studio兼容性问题
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
        merged_model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        print(f"✅ LoRA权重合并完成！")
        print(f"📁 合并后的模型已保存到: {output_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ 合并失败: {e}")
        return False

def main():
    print("🚀 LoRA权重合并工具")
    print("=" * 40)
    
    # 获取输入
    model_name = input("请输入模型名称 (例如: layue13/gemma3-1b-tool-use): ").strip()
    if not model_name:
        model_name = "layue13/gemma3-1b-tool-use"
    
    output_dir = input("请输入输出目录 (默认: ./gemma3-1b-tool-use-merged): ").strip()
    if not output_dir:
        output_dir = "./gemma3-1b-tool-use-merged"
    
    print(f"\n📊 合并配置:")
    print(f"  源模型: {model_name}")
    print(f"  输出目录: {output_dir}")
    
    # 执行合并
    print("\n" + "=" * 40)
    if merge_lora_weights(model_name, output_dir):
        print("\n🎉 合并成功！")
        print(f"📁 输出目录: {output_dir}")
        print("\n💡 现在可以:")
        print("1. 在LM Studio中加载合并后的模型")
        print("2. 使用convert_to_gguf.py转换为GGUF格式")
    else:
        print("\n❌ 合并失败")

if __name__ == "__main__":
    main()
