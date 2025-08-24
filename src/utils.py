"""
工具函数模块
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional
from pathlib import Path


def setup_logging(level: str = "INFO") -> None:
    """设置日志配置"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("training.log")
        ]
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """保存配置到文件"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def create_output_dir(output_dir: str) -> None:
    """创建输出目录"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)


def get_device_info() -> Dict[str, Any]:
    """获取设备信息"""
    import torch
    
    device_info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    if torch.cuda.is_available():
        device_info.update({
            "current_device": torch.cuda.current_device(),
            "device_name": torch.cuda.get_device_name(),
            "memory_allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
            "memory_reserved": torch.cuda.memory_reserved() / 1024**3,   # GB
        })
    
    return device_info


def format_tool_call_prompt(conversation: Dict[str, Any]) -> str:
    """格式化工具调用提示"""
    # 这里需要根据shawhin/tool-use-finetuning数据集的具体格式来实现
    # 暂时返回一个通用的格式
    messages = conversation.get("conversations", [])
    
    formatted_prompt = ""
    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")
        
        if role == "system":
            formatted_prompt += f"<|system|>\n{content}\n<|end|>\n"
        elif role == "user":
            formatted_prompt += f"<|user|>\n{content}\n<|end|>\n"
        elif role == "assistant":
            formatted_prompt += f"<|assistant|>\n{content}\n<|end|>\n"
        elif role == "tool":
            formatted_prompt += f"<|tool|>\n{content}\n<|end|>\n"
    
    return formatted_prompt


def validate_config(config: Dict[str, Any]) -> bool:
    """验证配置文件的完整性"""
    required_sections = ["model", "dataset", "lora", "training"]
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"配置文件缺少必需的 {section} 部分")
    
    # 验证模型配置
    if "name" not in config["model"]:
        raise ValueError("模型配置缺少 name 字段")
    
    # 验证数据集配置
    if "name" not in config["dataset"]:
        raise ValueError("数据集配置缺少 name 字段")
    
    # 验证LoRA配置
    required_lora_fields = ["r", "lora_alpha", "lora_dropout", "task_type"]
    for field in required_lora_fields:
        if field not in config["lora"]:
            raise ValueError(f"LoRA配置缺少必需的 {field} 字段")
    
    return True
