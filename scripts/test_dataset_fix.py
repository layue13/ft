#!/usr/bin/env python3
"""
测试数据集修复脚本
"""

import sys
import os

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils import setup_logging, load_config
from src.data_processor import ToolUseDataProcessor
from datasets import load_dataset


def test_dataset_loading():
    """测试数据集加载"""
    print("=== 测试数据集加载 ===")
    
    try:
        # 测试本地数据集加载
        train_dataset = load_dataset("arrow", data_files="data/train/data-*.arrow", split="train")
        print(f"✓ 本地训练集加载成功，样本数量: {len(train_dataset)}")
        print(f"  列名: {train_dataset.column_names}")
        
        # 显示第一个样本
        sample = train_dataset[0]
        print(f"  第一个样本:")
        for key, value in sample.items():
            if key == 'trace':
                print(f"    {key}: {type(value)} - 长度: {len(value)}")
                for i, item in enumerate(value[:2]):  # 只显示前2个
                    print(f"      [{i}]: {item}")
            else:
                print(f"    {key}: {type(value)} - {str(value)[:50]}...")
        
        return True
    except Exception as e:
        print(f"✗ 数据集加载失败: {e}")
        return False


def test_data_processing():
    """测试数据处理"""
    print("\n=== 测试数据处理 ===")
    
    try:
        # 加载配置
        config = load_config("configs/training_config_optimized.yaml")
        
        # 创建模拟tokenizer
        class MockTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return "mock_tokenized_text"
            
            def __call__(self, texts, **kwargs):
                return {
                    "input_ids": [[1, 2, 3, 4, 5]] * len(texts),
                    "attention_mask": [[1, 1, 1, 1, 1]] * len(texts)
                }
        
        mock_tokenizer = MockTokenizer()
        
        # 创建数据处理器
        processor = ToolUseDataProcessor(mock_tokenizer, config)
        
        # 测试对话格式化
        trace_sample = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！有什么可以帮助你的吗？"},
            {"role": "user", "content": "计算1+2"},
            {"role": "assistant", "content": {"name": "calculator", "arguments": {"a": 1, "b": 2}}}
        ]
        
        formatted = processor.format_conversation({"trace": trace_sample})
        print(f"✓ 对话格式化成功，消息数量: {len(formatted)}")
        
        # 测试分词函数
        examples = {
            "query": ["测试查询"],
            "trace": [trace_sample]
        }
        
        tokenized = processor.tokenize_function(examples)
        print(f"✓ 分词函数执行成功")
        print(f"  input_ids长度: {len(tokenized['input_ids'][0])}")
        print(f"  labels长度: {len(tokenized['labels'][0])}")
        
        return True
    except Exception as e:
        print(f"✗ 数据处理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_end_to_end():
    """端到端测试"""
    print("\n=== 端到端测试 ===")
    
    try:
        # 加载配置
        config = load_config("configs/training_config_optimized.yaml")
        
        # 创建模拟tokenizer
        class MockTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return "mock_tokenized_text"
            
            def __call__(self, texts, **kwargs):
                return {
                    "input_ids": [[1, 2, 3, 4, 5]] * len(texts),
                    "attention_mask": [[1, 1, 1, 1, 1]] * len(texts)
                }
        
        mock_tokenizer = MockTokenizer()
        
        # 创建数据处理器
        processor = ToolUseDataProcessor(mock_tokenizer, config)
        
        # 测试数据集处理
        train_dataset = load_dataset("arrow", data_files="data/train/data-*.arrow", split="train")
        print(f"✓ 原始数据集大小: {len(train_dataset)}")
        
        # 限制样本数量进行测试
        test_dataset = train_dataset.select(range(min(5, len(train_dataset))))
        processed_dataset = processor.process_dataset(test_dataset)
        
        print(f"✓ 数据集处理成功，处理后大小: {len(processed_dataset)}")
        
        return True
    except Exception as e:
        print(f"✗ 端到端测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    setup_logging("INFO")
    
    print("开始测试数据集修复...")
    
    # 测试数据集加载
    loading_ok = test_dataset_loading()
    
    # 测试数据处理
    processing_ok = test_data_processing()
    
    # 端到端测试
    e2e_ok = test_end_to_end()
    
    # 总结
    print("\n=== 测试总结 ===")
    if loading_ok and processing_ok and e2e_ok:
        print("✓ 所有测试通过！数据集修复成功。")
        print("\n现在可以运行训练脚本：")
        print("uv run python scripts/train.py --config configs/training_config_optimized.yaml")
    else:
        print("✗ 部分测试失败，需要进一步修复。")
    
    return loading_ok and processing_ok and e2e_ok


if __name__ == "__main__":
    main()
