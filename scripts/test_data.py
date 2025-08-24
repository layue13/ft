#!/usr/bin/env python3
"""
数据测试脚本
测试数据处理模块的功能
"""

import argparse
import logging
import sys
import os

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.utils import setup_logging, load_config
from src.model_config import load_model_and_tokenizer
from src.data_processor import create_data_processor


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="测试数据处理")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/training_config.yaml",
        help="配置文件路径"
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
    
    logger.info("开始数据测试...")
    
    try:
        # 加载配置
        config = load_config(args.config)
        
        # 加载模型和分词器
        logger.info("加载模型和分词器...")
        model, tokenizer = load_model_and_tokenizer(config)
        
        # 创建数据处理器
        logger.info("创建数据处理器...")
        data_processor = create_data_processor(tokenizer, config)
        
        # 测试数据格式化
        logger.info("测试数据格式化...")
        
        # 创建一个测试样本
        test_sample = {
            "query": "whats some simple steps to study better for school?",
            "query_type": "no_tool",
            "trace": [
                {
                    "content": "<instructions>\nYou are an AI assistant with access to a set of tools...",
                    "role": "user"
                },
                {
                    "content": "whats some simple steps to study better for school?",
                    "role": "user"
                },
                {
                    "content": "<final_answer>\nOkay, here's a breakdown of simple steps...",
                    "role": "assistant"
                }
            ],
            "num_tools_available": 1,
            "tool_needed": False,
            "tool_name": None
        }
        
        # 测试格式化
        formatted = data_processor.format_conversation(test_sample)
        logger.info("格式化结果:")
        logger.info(formatted[:500] + "..." if len(formatted) > 500 else formatted)
        
        # 测试分词
        logger.info("测试分词...")
        tokenized = data_processor.tokenize_function({
            "query": [test_sample["query"]],
            "trace": [test_sample["trace"]]
        })
        
        logger.info(f"分词结果: input_ids长度={len(tokenized['input_ids'][0])}")
        logger.info(f"解码结果: {tokenizer.decode(tokenized['input_ids'][0][:100])}...")
        
        logger.info("数据测试完成!")
        
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        raise


if __name__ == "__main__":
    main()
