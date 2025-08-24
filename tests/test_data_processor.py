"""
数据处理模块测试 - 优化版本
"""

import pytest
import json
from unittest.mock import Mock, patch
from src.data_processor import ToolUseDataProcessor, ToolCallValidator


class TestToolCallValidator:
    """工具调用验证器测试"""
    
    def test_validate_tool_call_valid(self):
        """测试有效的工具调用格式"""
        valid_tool_call = {
            "name": "calculator",
            "arguments": {"a": 1, "b": 2}
        }
        
        is_valid, message = ToolCallValidator.validate_tool_call(valid_tool_call)
        assert is_valid
        assert message == "格式正确"
    
    def test_validate_tool_call_invalid_type(self):
        """测试无效的工具调用类型"""
        invalid_tool_call = "not a dict"
        
        is_valid, message = ToolCallValidator.validate_tool_call(invalid_tool_call)
        assert not is_valid
        assert "字典格式" in message
    
    def test_validate_tool_call_missing_name(self):
        """测试缺少工具名称"""
        invalid_tool_call = {
            "arguments": {"a": 1, "b": 2}
        }
        
        is_valid, message = ToolCallValidator.validate_tool_call(invalid_tool_call)
        assert not is_valid
        assert "name" in message
    
    def test_validate_tool_call_missing_arguments(self):
        """测试缺少工具参数"""
        invalid_tool_call = {
            "name": "calculator"
        }
        
        is_valid, message = ToolCallValidator.validate_tool_call(invalid_tool_call)
        assert not is_valid
        assert "arguments" in message
    
    def test_extract_tool_calls(self):
        """测试从文本中提取工具调用"""
        text = """
        <tool_call>
        name: calculator
        arguments: {"a": 1, "b": 2}
        result: 3
        </tool_call>
        """
        
        tool_calls = ToolCallValidator.extract_tool_calls(text)
        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "calculator"
        assert tool_calls[0]["arguments"] == {"a": 1, "b": 2}
        assert tool_calls[0]["result"] == "3"
    
    def test_extract_tool_calls_multiple(self):
        """测试提取多个工具调用"""
        text = """
        <tool_call>
        name: calculator
        arguments: {"a": 1, "b": 2}
        </tool_call>
        <tool_call>
        name: weather
        arguments: {"city": "Beijing"}
        </tool_call>
        """
        
        tool_calls = ToolCallValidator.extract_tool_calls(text)
        assert len(tool_calls) == 2
        assert tool_calls[0]["name"] == "calculator"
        assert tool_calls[1]["name"] == "weather"


class TestToolUseDataProcessor:
    """工具调用数据处理器测试 - 优化版本"""
    
    def setup_method(self):
        """设置测试环境"""
        self.mock_tokenizer = Mock()
        self.mock_tokenizer.pad_token = "<pad>"
        self.mock_tokenizer.eos_token = "</s>"
        
        self.config = {
            "data_processing": {
                "max_seq_length": 512,
                "remove_columns": ["id"]
            }
        }
        
        self.processor = ToolUseDataProcessor(self.mock_tokenizer, self.config)
    
    def test_format_tool_call_valid(self):
        """测试格式化有效工具调用"""
        tool_content = {
            "name": "calculator",
            "arguments": {"a": 1, "b": 2},
            "result": "3"
        }
        
        formatted = self.processor.format_tool_call(tool_content)
        assert "<tool_call>" in formatted
        assert "name: calculator" in formatted
        assert '"a": 1' in formatted
        assert '"b": 2' in formatted
        assert "result: 3" in formatted
    
    def test_format_tool_call_invalid(self):
        """测试格式化无效工具调用"""
        tool_content = {
            "name": "calculator"
            # 缺少arguments
        }
        
        formatted = self.processor.format_tool_call(tool_content)
        assert "<tool_call>" in formatted
        assert "name: calculator" in formatted
    
    def test_format_conversation_simple(self):
        """测试简单对话格式化"""
        conversation = {
            "conversations": [
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "你好！有什么可以帮助你的吗？"}
            ]
        }
        
        result = self.processor.format_conversation(conversation)
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"
    
    def test_format_conversation_with_tool_call(self):
        """测试包含工具调用的对话格式化"""
        conversation = {
            "conversations": [
                {"role": "user", "content": "计算1+2"},
                {"role": "assistant", "content": {
                    "name": "calculator",
                    "arguments": {"a": 1, "b": 2}
                }}
            ]
        }
        
        result = self.processor.format_conversation(conversation)
        assert len(result) == 2
        assert result[1]["role"] == "assistant"
        assert "<tool_call>" in result[1]["content"][0]["text"]
    
    def test_tokenize_function(self):
        """测试分词函数"""
        examples = {
            "query": ["Hello world", "Test message"],
            "trace": [
                [{"role": "user", "content": "Hello"}],
                [{"role": "user", "content": "Test"}]
            ]
        }
        
        self.mock_tokenizer.apply_chat_template.return_value = {
            "input_ids": [1, 2, 3],
            "attention_mask": [1, 1, 1]
        }
        
        result = self.processor.tokenize_function(examples)
        
        assert "input_ids" in result
        assert "attention_mask" in result
        assert "labels" in result
        assert result["labels"] == result["input_ids"]
    
    def test_tokenize_function_fallback(self):
        """测试分词函数回退机制"""
        examples = {
            "query": ["Hello world"],
            "trace": [[{"role": "user", "content": "Hello"}]]
        }
        
        # 模拟聊天模板失败
        self.mock_tokenizer.apply_chat_template.side_effect = Exception("Template error")
        self.mock_tokenizer.return_value = {
            "input_ids": [1, 2, 3],
            "attention_mask": [1, 1, 1]
        }
        
        result = self.processor.tokenize_function(examples)
        
        assert "input_ids" in result
        assert "attention_mask" in result
        assert "labels" in result
    
    def test_filter_dataset_with_limit(self):
        """测试数据集过滤（有限制）"""
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
    
    @patch('src.data_processor.load_dataset')
    def test_load_dataset_with_mirror(self, mock_load_dataset):
        """测试使用镜像站加载数据集"""
        mock_dataset = Mock()
        mock_load_dataset.return_value = mock_dataset
        
        self.config["mirror"] = {"name": "hf_mirror"}
        
        result = self.processor.load_dataset("test_dataset", "train")
        
        assert result == mock_dataset
        mock_load_dataset.assert_called_once()
    
    @patch('src.data_processor.load_dataset')
    def test_load_dataset_without_mirror(self, mock_load_dataset):
        """测试不使用镜像站加载数据集"""
        mock_dataset = Mock()
        mock_load_dataset.return_value = mock_dataset
        
        result = self.processor.load_dataset("test_dataset", "train")
        
        assert result == mock_dataset
        mock_load_dataset.assert_called_once_with("test_dataset", split="train")


class TestDataProcessorIntegration:
    """数据处理器集成测试"""
    
    def test_end_to_end_processing(self):
        """测试端到端数据处理"""
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "<pad>"
        mock_tokenizer.eos_token = "</s>"
        mock_tokenizer.apply_chat_template.return_value = {
            "input_ids": [1, 2, 3],
            "attention_mask": [1, 1, 1]
        }
        
        config = {
            "data_processing": {
                "max_seq_length": 512,
                "remove_columns": ["id"]
            }
        }
        
        processor = ToolUseDataProcessor(mock_tokenizer, config)
        
        # 测试工具调用验证
        valid_tool = {"name": "test", "arguments": {"param": "value"}}
        is_valid, _ = processor.validator.validate_tool_call(valid_tool)
        assert is_valid
        
        # 测试工具调用格式化
        formatted = processor.format_tool_call(valid_tool)
        assert "<tool_call>" in formatted
        assert "name: test" in formatted
