#!/usr/bin/env python3
"""
检查数据集结构脚本
"""

import sys
import os

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from datasets import load_dataset
from src.utils import setup_logging, load_config


def check_dataset_structure():
    """检查数据集结构"""
    setup_logging("INFO")
    
    # 加载数据集
    print("=== 检查数据集结构 ===")
    
    try:
        # 加载训练集
        train_dataset = load_dataset("shawhin/tool-use-finetuning", split="train")
        print(f"训练集大小: {len(train_dataset)}")
        print(f"训练集列名: {train_dataset.column_names}")
        
        # 显示第一个样本
        if len(train_dataset) > 0:
            print("\n=== 第一个训练样本 ===")
            sample = train_dataset[0]
            for key, value in sample.items():
                print(f"{key}: {type(value)} - {str(value)[:100]}...")
        
        # 加载验证集
        eval_dataset = load_dataset("shawhin/tool-use-finetuning", split="validation")
        print(f"\n验证集大小: {len(eval_dataset)}")
        print(f"验证集列名: {eval_dataset.column_names}")
        
        # 显示第一个验证样本
        if len(eval_dataset) > 0:
            print("\n=== 第一个验证样本 ===")
            sample = eval_dataset[0]
            for key, value in sample.items():
                print(f"{key}: {type(value)} - {str(value)[:100]}...")
        
        # 分析trace字段
        print("\n=== 分析trace字段 ===")
        if 'trace' in train_dataset.column_names:
            trace_sample = train_dataset[0]['trace']
            print(f"trace类型: {type(trace_sample)}")
            if isinstance(trace_sample, list):
                print(f"trace长度: {len(trace_sample)}")
                for i, item in enumerate(trace_sample):
                    print(f"  trace[{i}]: {type(item)} - {item}")
        
        return True
        
    except Exception as e:
        print(f"检查数据集时出错: {e}")
        return False


def main():
    """主函数"""
    success = check_dataset_structure()
    
    if success:
        print("\n=== 数据集检查完成 ===")
        print("现在可以运行训练脚本了")
    else:
        print("\n=== 数据集检查失败 ===")
        sys.exit(1)


if __name__ == "__main__":
    main()
