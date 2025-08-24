"""
数据处理模块测试
"""

import pytest
from unittest.mock import Mock, patch
from src.data_processor import ToolUseDataProcessor


class TestToolUseDataProcessor:
    """工具调用数据处理器测试"""
    
    def setup_method(self):
        """设置测试环境"""
        self.mock_tokenizer = Mock()
        self.mock_tokenizer.pad_token = "<pad>"
        self.mock_tokenizer.eos_token = "</s>"
        
        self.config = {
            "data_processing": {
                "max_seq_length": 512,
                "text_column": "text",
                "remove_columns": ["id"],
                "template_format": "tool_use"
            }
        }
        
        self.processor = ToolUseDataProcessor(self.mock_tokenizer, self.config)
    
    def test_format_conversation(self):
        """测试对话格式化"""
        conversation = {
            "conversations": [
                {"role": "system", "content": "你是一个助手"},
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "你好！有什么可以帮助你的吗？"},
                {"role": "tool", "content": {"name": "calculator", "arguments": {"a": 1, "b": 2}}}
            ]
        }
        
        result = self.processor.format_conversation(conversation)
        
        assert "<|system|>" in result
        assert "<|user|>" in result
        assert "<|assistant|>" in result
        assert "<|tool|>" in result
        assert "calculator" in result
    
    def test_tokenize_function(self):
        """测试分词函数"""
        examples = {
            "text": ["Hello world", "Test message"]
        }
        
        self.mock_tokenizer.return_value = {
            "input_ids": [[1, 2, 3], [4, 5, 6]],
            "attention_mask": [[1, 1, 1], [1, 1, 1]]
        }
        
        result = self.processor.tokenize_function(examples)
        
        assert "input_ids" in result
        assert "attention_mask" in result
        assert "labels" in result
        assert result["labels"] == result["input_ids"]
    
    def test_filter_dataset(self):
        """测试数据集过滤"""
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=100)
        mock_dataset.select = Mock(return_value=mock_dataset)
        
        result = self.processor.filter_dataset(mock_dataset, max_samples=50)
        
        mock_dataset.select.assert_called_once_with(range(50))
    
    def test_filter_dataset_no_limit(self):
        """测试数据集过滤（无限制）"""
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=100)
        
        result = self.processor.filter_dataset(mock_dataset, max_samples=None)
        
        assert result == mock_dataset
