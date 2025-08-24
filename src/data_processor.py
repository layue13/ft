"""
数据处理模块 - 优化版本
"""

import json
import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizer


logger = logging.getLogger(__name__)


class ToolCallValidator:
    """工具调用格式验证器"""
    
    @staticmethod
    def validate_tool_call(content: Any) -> Tuple[bool, str]:
        """验证工具调用格式"""
        if not isinstance(content, dict):
            return False, "工具调用内容必须是字典格式"
        
        required_fields = ["name", "arguments"]
        for field in required_fields:
            if field not in content:
                return False, f"缺少必需字段: {field}"
        
        if not isinstance(content["name"], str):
            return False, "工具名称必须是字符串"
        
        if not isinstance(content["arguments"], dict):
            return False, "工具参数必须是字典格式"
        
        return True, "格式正确"
    
    @staticmethod
    def extract_tool_calls(text: str) -> List[Dict[str, Any]]:
        """从文本中提取工具调用"""
        tool_calls = []
        # 匹配 <tool_call>...</tool_call> 格式
        pattern = r'<tool_call>(.*?)</tool_call>'
        matches = re.findall(pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                # 解析工具调用内容
                lines = match.strip().split('\n')
                tool_call = {}
                
                for line in lines:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        if key == 'name':
                            tool_call['name'] = value
                        elif key == 'arguments':
                            tool_call['arguments'] = json.loads(value)
                        elif key == 'result':
                            tool_call['result'] = value
                
                if 'name' in tool_call and 'arguments' in tool_call:
                    tool_calls.append(tool_call)
            except Exception as e:
                logger.warning(f"解析工具调用失败: {e}")
        
        return tool_calls


class ToolUseDataProcessor:
    """工具调用数据处理器 - 优化版本"""
    
    def __init__(self, tokenizer: PreTrainedTokenizer, config: Dict[str, Any]):
        self.tokenizer = tokenizer
        self.config = config
        self.max_length = config["data_processing"]["max_seq_length"]
        self.validator = ToolCallValidator()
        self._template_warning_count = 0  # 添加警告计数器
        
    def load_dataset(self, dataset_name: str, split: str = "train") -> Dataset:
        """加载数据集"""
        logger.info(f"正在加载数据集: {dataset_name}, 分割: {split}")
        
        # 获取镜像站配置
        mirror_name = self.config.get("mirror", {}).get("name")
        if mirror_name:
            from .mirror_utils import get_mirror_selector
            selector = get_mirror_selector()
            dataset_url = selector.get_dataset_url(dataset_name, mirror_name)
            logger.info(f"使用镜像站 {mirror_name} 加载数据集: {dataset_url}")
        else:
            dataset_url = dataset_name
            logger.info(f"使用官方源加载数据集: {dataset_url}")
        
        try:
            dataset = load_dataset(dataset_url, split=split)
            logger.info(f"成功加载数据集，样本数量: {len(dataset)}")
            return dataset
        except Exception as e:
            logger.error(f"加载数据集失败: {e}")
            raise
    
    def format_tool_call(self, tool_content: Dict[str, Any]) -> str:
        """格式化工具调用 - 简化版本"""
        tool_name = tool_content.get("name", "")
        tool_args = tool_content.get("arguments", {})
        tool_result = tool_content.get("result", "")
        
        # 验证工具调用格式
        is_valid, message = self.validator.validate_tool_call(tool_content)
        if not is_valid:
            logger.warning(f"工具调用格式无效: {message}")
            return f"<tool_call>\nname: {tool_name}\narguments: {json.dumps(tool_args, ensure_ascii=False)}\n</tool_call>"
        
        # 构建标准格式
        formatted = f"<tool_call>\nname: {tool_name}\narguments: {json.dumps(tool_args, ensure_ascii=False)}"
        if tool_result:
            formatted += f"\nresult: {tool_result}"
        formatted += "\n</tool_call>"
        
        return formatted
    
    def format_conversation(self, conversation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """格式化对话 - 优化版本，确保角色交替"""
        # 获取消息列表
        messages = conversation.get("trace", conversation.get("conversations", []))
        if not messages:
            return []
        
        gemma_messages = []
        last_role = None
        
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            
            # 跳过system角色，因为它不参与交替
            if role == "system":
                gemma_messages.append({
                    "role": "system",
                    "content": [{"type": "text", "text": content}]
                })
                continue
            
            # 确保user和assistant角色交替
            if role == "user":
                if last_role == "user":
                    # 如果连续两个user，添加一个空的assistant消息
                    gemma_messages.append({
                        "role": "assistant",
                        "content": [{"type": "text", "text": ""}]
                    })
                gemma_messages.append({
                    "role": "user", 
                    "content": [{"type": "text", "text": content}]
                })
                last_role = "user"
            elif role == "assistant":
                if last_role == "assistant":
                    # 如果连续两个assistant，添加一个空的user消息
                    gemma_messages.append({
                        "role": "user",
                        "content": [{"type": "text", "text": ""}]
                    })
                
                # 检查是否包含工具调用
                if isinstance(content, dict) and "name" in content:
                    # 工具调用格式
                    tool_text = self.format_tool_call(content)
                    gemma_messages.append({
                        "role": "assistant",
                        "content": [{"type": "text", "text": tool_text}]
                    })
                else:
                    # 普通文本
                    gemma_messages.append({
                        "role": "assistant",
                        "content": [{"type": "text", "text": content}]
                    })
                last_role = "assistant"
            elif role == "tool":
                # 工具结果，作为assistant消息处理
                if last_role == "assistant":
                    # 如果连续两个assistant，添加一个空的user消息
                    gemma_messages.append({
                        "role": "user",
                        "content": [{"type": "text", "text": ""}]
                    })
                
                if isinstance(content, dict):
                    tool_text = self.format_tool_call(content)
                    gemma_messages.append({
                        "role": "assistant",
                        "content": [{"type": "text", "text": tool_text}]
                    })
                else:
                    gemma_messages.append({
                        "role": "assistant",
                        "content": [{"type": "text", "text": content}]
                    })
                last_role = "assistant"
        
        return gemma_messages
    
    def tokenize_function(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        """分词函数 - 优化版本"""
        texts = []
        
        for i in range(len(examples.get("query", []))):
            query = examples["query"][i]
            trace = examples["trace"][i]
            
            # 格式化对话
            gemma_messages = self.format_conversation({"trace": trace})
            
            # 添加用户查询
            gemma_messages.append({
                "role": "user",
                "content": [{"type": "text", "text": query}]
            })
            
            # 使用Gemma3聊天模板
            try:
                inputs = self.tokenizer.apply_chat_template(
                    gemma_messages,
                    tokenize=True,
                    return_dict=True,
                    return_tensors=None,
                    add_generation_prompt=True,
                )
                texts.append(inputs)
            except Exception as e:
                # 限制警告日志的输出频率
                self._template_warning_count += 1
                if self._template_warning_count <= 5:  # 只显示前5个警告
                    logger.warning(f"聊天模板处理失败，使用简单格式: {e}")
                elif self._template_warning_count == 6:
                    logger.warning("聊天模板处理失败次数过多，后续警告将不再显示...")
                
                # 回退到简单格式
                simple_text = f"User: {query}\nAssistant: "
                texts.append(simple_text)
        
        # 分词处理
        if isinstance(texts[0], dict):
            tokenized = texts
        else:
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                padding=False,
                max_length=self.max_length,
                return_tensors=None,
            )
        
        # 设置labels
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    def process_dataset(self, dataset: Dataset) -> Dataset:
        """处理数据集 - 优化版本"""
        logger.info("开始处理数据集...")
        logger.info(f"原始数据集大小: {len(dataset)}")
        
        # 移除不需要的列
        remove_columns = self.config["data_processing"].get("remove_columns", [])
        if remove_columns:
            logger.info(f"移除列: {remove_columns}")
            dataset = dataset.remove_columns(remove_columns)
        
        # 应用分词
        logger.info("开始分词处理...")
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="分词处理中...",
        )
        
        logger.info(f"数据集处理完成，样本数量: {len(tokenized_dataset)}")
        
        # 打印样本示例
        if len(tokenized_dataset) > 0:
            sample = tokenized_dataset[0]
            logger.info("=== 样本示例 ===")
            logger.info(f"input_ids长度: {len(sample['input_ids'])}")
            logger.info(f"attention_mask长度: {len(sample['attention_mask'])}")
            logger.info(f"labels长度: {len(sample['labels'])}")
            logger.info("================")
        
        return tokenized_dataset
    
    def filter_dataset(self, dataset: Dataset, max_samples: Optional[int] = None) -> Dataset:
        """过滤数据集"""
        if max_samples and len(dataset) > max_samples:
            logger.info(f"限制数据集大小从 {len(dataset)} 到 {max_samples}")
            dataset = dataset.select(range(max_samples))
        
        return dataset
    
    def prepare_training_data(self, dataset_name: str) -> Dict[str, Dataset]:
        """准备训练数据 - 优化版本"""
        dataset_config = self.config["dataset"]
        
        # 加载训练集
        train_dataset = self.load_dataset(
            dataset_name, 
            dataset_config["train_split"]
        )
        
        # 加载验证集
        try:
            eval_dataset = self.load_dataset(
                dataset_name, 
                dataset_config["validation_split"]
            )
        except Exception as e:
            logger.warning(f"无法加载验证集: {e}")
            # 从训练集中分割
            split_dataset = train_dataset.train_test_split(test_size=0.1)
            train_dataset = split_dataset["train"]
            eval_dataset = split_dataset["test"]
        
        # 过滤数据集
        max_samples = dataset_config.get("max_samples")
        train_dataset = self.filter_dataset(train_dataset, max_samples)
        eval_dataset = self.filter_dataset(eval_dataset, max_samples)
        
        # 处理数据集
        train_dataset = self.process_dataset(train_dataset)
        eval_dataset = self.process_dataset(eval_dataset)
        
        return {
            "train": train_dataset,
            "eval": eval_dataset
        }


def create_data_processor(tokenizer: PreTrainedTokenizer, config: Dict[str, Any]) -> ToolUseDataProcessor:
    """创建数据处理器"""
    return ToolUseDataProcessor(tokenizer, config)
