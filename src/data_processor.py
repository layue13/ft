"""
数据处理模块
"""

import json
import logging
from typing import Dict, Any, List, Optional
from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizer


logger = logging.getLogger(__name__)


class ToolUseDataProcessor:
    """工具调用数据处理器"""
    
    def __init__(self, tokenizer: PreTrainedTokenizer, config: Dict[str, Any]):
        self.tokenizer = tokenizer
        self.config = config
        self.max_length = config["data_processing"]["max_seq_length"]
        self.text_column = config["data_processing"]["text_column"]
        self.remove_columns = config["data_processing"]["remove_columns"]
        self.template_format = config["data_processing"]["template_format"]
    
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
    
    def format_conversation(self, conversation: Dict[str, Any]) -> list:
        """格式化对话为Gemma3消息格式"""
        # 处理shawhin/tool-use-finetuning数据集的格式
        if "trace" in conversation:
            messages = conversation["trace"]
        else:
            messages = conversation.get("conversations", [])
        
        if not messages:
            return []
        
        # 转换为Gemma3聊天模板格式
        gemma_messages = []
        
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            
            if role == "system":
                gemma_messages.append({
                    "role": "system",
                    "content": [{"type": "text", "text": content}]
                })
            elif role == "user":
                gemma_messages.append({
                    "role": "user", 
                    "content": [{"type": "text", "text": content}]
                })
            elif role == "assistant":
                gemma_messages.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": content}]
                })
            elif role == "tool":
                # 工具调用格式
                if isinstance(content, dict):
                    tool_name = content.get("name", "")
                    tool_args = content.get("arguments", {})
                    tool_result = content.get("result", "")
                    
                    tool_content = f"<tool_call>\n"
                    tool_content += f"name: {tool_name}\n"
                    tool_content += f"arguments: {json.dumps(tool_args, ensure_ascii=False)}\n"
                    if tool_result:
                        tool_content += f"result: {tool_result}\n"
                    tool_content += f"</tool_call>"
                    
                    gemma_messages.append({
                        "role": "assistant",
                        "content": [{"type": "text", "text": tool_content}]
                    })
                else:
                    gemma_messages.append({
                        "role": "assistant",
                        "content": [{"type": "text", "text": content}]
                    })
        
        return gemma_messages
    
    def tokenize_function(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        """分词函数"""
        # 处理shawhin/tool-use-finetuning数据集格式
        texts = []
        for i in range(len(examples.get("query", []))):
            # 构建完整的对话文本
            query = examples["query"][i]
            trace = examples["trace"][i]
            
            # 格式化对话为Gemma3消息格式
            gemma_messages = self.format_conversation({"trace": trace})
            
            # 添加用户查询
            gemma_messages.append({
                "role": "user",
                "content": [{"type": "text", "text": query}]
            })
            
            # 使用Gemma3的聊天模板
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
                # 如果聊天模板失败，回退到简单格式
                logger.warning(f"聊天模板处理失败，使用简单格式: {e}")
                simple_text = f"User: {query}\nAssistant: "
                texts.append(simple_text)
        
        # 分词
        if isinstance(texts[0], dict):
            # 如果已经是tokenized格式
            tokenized = texts
        else:
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                padding=False,
                max_length=self.max_length,
                return_tensors=None,
            )
        
        # 设置labels为input_ids的副本
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    def process_dataset(self, dataset: Dataset) -> Dataset:
        """处理数据集"""
        logger.info("开始处理数据集...")
        logger.info(f"原始数据集大小: {len(dataset)}")
        logger.info(f"原始数据集列名: {dataset.column_names}")
        
        # 移除不需要的列
        if self.remove_columns:
            logger.info(f"移除列: {self.remove_columns}")
            dataset = dataset.remove_columns(self.remove_columns)
            logger.info(f"移除列后数据集列名: {dataset.column_names}")
        
        # 应用分词
        logger.info("开始分词处理...")
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="分词处理中...",
        )
        
        logger.info(f"数据集处理完成，样本数量: {len(tokenized_dataset)}")
        logger.info(f"处理后数据集列名: {tokenized_dataset.column_names}")
        
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
        """准备训练数据"""
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
            # 如果没有验证集，从训练集中分割
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
