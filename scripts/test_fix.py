#!/usr/bin/env python3
"""
测试修复脚本
验证数据处理和配置修复是否有效
"""

import sys
import os

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils import setup_logging, load_config, validate_config
from src.model_config import create_training_arguments


def test_config_validation():
    """测试配置验证"""
    print("=== 测试配置验证 ===")
    
    try:
        # 加载配置
        config = load_config("configs/training_config_optimized.yaml")
        print("✓ 配置加载成功")
        
        # 验证配置
        validate_config(config)
        print("✓ 配置验证通过")
        
        # 测试训练参数创建
        training_args = create_training_arguments(config)
        print("✓ 训练参数创建成功")
        print(f"  输出目录: {training_args.output_dir}")
        print(f"  学习率: {training_args.learning_rate}")
        print(f"  训练轮数: {training_args.num_train_epochs}")
        
        return True
    except Exception as e:
        print(f"✗ 配置测试失败: {e}")
        return False


def test_data_processing():
    """测试数据处理"""
    print("\n=== 测试数据处理 ===")
    
    try:
        from src.data_processor import ToolUseDataProcessor, ToolCallValidator
        
        # 测试工具调用验证
        validator = ToolCallValidator()
        valid_tool = {"name": "test", "arguments": {"param": "value"}}
        is_valid, message = validator.validate_tool_call(valid_tool)
        print(f"✓ 工具调用验证: {is_valid}, {message}")
        
        # 测试对话格式化
        conversation = {
            "trace": [
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "你好！有什么可以帮助你的吗？"},
                {"role": "user", "content": "计算1+2"},
                {"role": "assistant", "content": {"name": "calculator", "arguments": {"a": 1, "b": 2}}}
            ]
        }
        
        # 创建模拟tokenizer
        class MockTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}
        
        mock_tokenizer = MockTokenizer()
        config = {"data_processing": {"max_seq_length": 512}}
        
        processor = ToolUseDataProcessor(mock_tokenizer, config)
        formatted = processor.format_conversation(conversation)
        print(f"✓ 对话格式化成功，消息数量: {len(formatted)}")
        
        return True
    except Exception as e:
        print(f"✗ 数据处理测试失败: {e}")
        return False


def main():
    """主函数"""
    setup_logging("INFO")
    
    print("开始测试修复...")
    
    # 测试配置
    config_ok = test_config_validation()
    
    # 测试数据处理
    data_ok = test_data_processing()
    
    # 总结
    print("\n=== 测试总结 ===")
    if config_ok and data_ok:
        print("✓ 所有测试通过！修复成功。")
        print("\n现在可以运行训练脚本：")
        print("uv run python scripts/train.py --config configs/training_config_optimized.yaml")
    else:
        print("✗ 部分测试失败，需要进一步修复。")
    
    return config_ok and data_ok


if __name__ == "__main__":
    main()
