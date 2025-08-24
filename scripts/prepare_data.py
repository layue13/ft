#!/usr/bin/env python3
"""
数据准备脚本
下载和预处理shawhin/tool-use-finetuning数据集
"""

import argparse
import logging
import sys
import os
from datasets import load_dataset

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.utils import setup_logging


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="准备工具调用数据集")
    parser.add_argument(
        "--dataset_name", 
        type=str, 
        default="shawhin/tool-use-finetuning",
        help="数据集名称"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./data",
        help="输出目录"
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
    
    logger.info("开始数据准备流程...")
    
    try:
        # 创建输出目录
        os.makedirs(args.output_dir, exist_ok=True)
        
        # 加载数据集
        logger.info(f"加载数据集: {args.dataset_name}")
        dataset = load_dataset(args.dataset_name)
        
        logger.info(f"数据集信息: {dataset}")
        
        # 显示数据集结构
        for split_name, split_data in dataset.items():
            logger.info(f"分割 '{split_name}': {len(split_data)} 样本")
            if len(split_data) > 0:
                logger.info(f"样本结构: {split_data[0]}")
        
        # 保存数据集到本地
        logger.info(f"保存数据集到: {args.output_dir}")
        dataset.save_to_disk(args.output_dir)
        
        logger.info("数据准备完成!")
        
    except Exception as e:
        logger.error(f"数据准备过程中发生错误: {e}")
        raise


if __name__ == "__main__":
    main()
