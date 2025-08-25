#!/usr/bin/env python3
"""
测试模型加载脚本
验证模型和分词器加载是否正常工作
"""

import sys
import os
import logging
import argparse

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils import setup_logging, load_config
from src.model_config import load_model_and_tokenizer


def test_model_loading(config_path: str = "configs/training_config_optimized.yaml"):
    """测试模型加载"""
    print("=== 测试模型加载 ===")
    
    try:
        # 设置日志
        setup_logging("INFO")
        logger = logging.getLogger(__name__)
        
        # 加载配置
        config = load_config(config_path)
        logger.info("配置加载成功")
        
        # 加载模型和分词器
        logger.info("开始加载模型和分词器...")
        model, tokenizer = load_model_and_tokenizer(config)
        
        logger.info("✓ 模型和分词器加载成功!")
        logger.info(f"  模型类型: {type(model).__name__}")
        logger.info(f"  分词器类型: {type(tokenizer).__name__}")
        logger.info(f"  词汇表大小: {tokenizer.vocab_size}")
        
        # 测试分词
        test_text = "Hello, how can I help you today?"
        tokens = tokenizer(test_text, return_tensors="pt")
        logger.info(f"  测试分词成功: {test_text}")
        logger.info(f"  Token数量: {tokens.input_ids.shape[1]}")
        
        print("✓ 所有测试通过！模型可以正常使用。")
        return True
        
    except Exception as e:
        print(f"✗ 模型加载测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="测试模型加载")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config_optimized.yaml",
        help="配置文件路径"
    )
    args = parser.parse_args()
    
    print("开始测试模型加载...")
    
    success = test_model_loading(args.config)
    
    if success:
        print("\n=== 测试成功 ===")
        print("现在可以运行训练脚本：")
        print(f"uv run python scripts/train.py --config {args.config}")
    else:
        print("\n=== 测试失败 ===")
        print("请检查错误信息并修复问题。")
    
    return success


if __name__ == "__main__":
    main()