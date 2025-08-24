#!/usr/bin/env python3
"""
评估脚本
评估微调后的模型性能
"""

import argparse
import logging
import sys
import os
import torch
from transformers import AutoTokenizer
from peft import PeftModel, PeftConfig

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.utils import setup_logging, load_config


def load_finetuned_model(model_path: str):
    """加载微调后的模型"""
    # 加载PEFT配置
    peft_config = PeftConfig.from_pretrained(model_path)
    
    # 加载基础模型
    from transformers import AutoModelForCausalLM
    base_model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # 加载PEFT适配器
    model = PeftModel.from_pretrained(base_model, model_path)
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, max_length: int = 512):
    """生成回复"""
    # 编码输入
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
        padding=True
    )
    
    # 移动到GPU
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
        model = model.cuda()
    
    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # 解码输出
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="评估微调后的模型")
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="模型路径"
    )
    parser.add_argument(
        "--test_prompts", 
        type=str, 
        nargs="+",
        default=[
            "请帮我查询今天的天气",
            "计算 123 + 456 的结果",
            "翻译 'Hello, world!' 为中文"
        ],
        help="测试提示"
    )
    parser.add_argument(
        "--max_length", 
        type=int, 
        default=512,
        help="最大生成长度"
    )
    parser.add_argument(
        "--log_level", 
        type=str, 
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别"
    )
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("开始模型评估...")
    
    try:
        # 加载模型
        logger.info(f"加载模型: {args.model_path}")
        model, tokenizer = load_finetuned_model(args.model_path)
        
        # 设置模型为评估模式
        model.eval()
        
        logger.info("模型加载完成!")
        
        # 测试生成
        logger.info("开始测试生成...")
        
        for i, prompt in enumerate(args.test_prompts, 1):
            logger.info(f"\n测试 {i}: {prompt}")
            logger.info("-" * 50)
            
            try:
                response = generate_response(model, tokenizer, prompt, args.max_length)
                logger.info(f"模型回复:\n{response}")
            except Exception as e:
                logger.error(f"生成失败: {e}")
            
            logger.info("-" * 50)
        
        logger.info("评估完成!")
        
    except Exception as e:
        logger.error(f"评估过程中发生错误: {e}")
        raise


if __name__ == "__main__":
    main()
