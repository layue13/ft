#!/usr/bin/env python3
"""
测试模型脚本 - 验证工具调用功能
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def test_model():
    print("🚀 开始测试模型...")
    
    model_path = "./gemma3-1b-tool-use-merged"
    
    try:
        print("📦 加载模型...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        print("✅ 模型加载成功！")
        
        # 测试用例
        test_cases = [
            "帮我查询北京的天气",
            "请帮我计算23+45",
            "我想知道今天的日期",
            "帮我搜索Python教程"
        ]
        
        print("\n🧪 开始测试工具调用...")
        
        for i, prompt in enumerate(test_cases, 1):
            print(f"\n--- 测试 {i}: {prompt} ---")
            
            # 格式化输入
            formatted_prompt = f"<bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
            
            # 编码
            inputs = tokenizer(formatted_prompt, return_tensors="pt")
            
            # 生成
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=inputs['input_ids'].shape[1] + 200,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # 解码
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取模型回复部分
            if "<start_of_turn>model" in response:
                model_response = response.split("<start_of_turn>model")[1].split("<end_of_turn>")[0].strip()
                print(f"助手: {model_response}")
            else:
                print(f"完整回复: {response}")
        
        print("\n🎉 测试完成！")
        print("💡 如果看到<tool_call>格式的回复，说明模型训练成功！")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        print("💡 请检查模型文件是否完整")

if __name__ == "__main__":
    test_model()
