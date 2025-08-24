"""
工具函数模块 - 优化版本
"""

import logging
import os
import yaml
import torch
from typing import Dict, Any, Optional


logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO") -> None:
    """设置日志配置"""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training.log', encoding='utf-8')
        ]
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 修复科学计数法解析问题
    if "training" in config:
        training = config["training"]
        if "learning_rate" in training:
            # 确保学习率是数值类型
            lr = training["learning_rate"]
            if isinstance(lr, str):
                if "e" in lr.lower():
                    # 处理科学计数法
                    training["learning_rate"] = float(lr)
                else:
                    training["learning_rate"] = float(lr)
    
    return config


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """保存配置文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def validate_config(config: Dict[str, Any]) -> None:
    """验证配置文件 - 新增验证逻辑"""
    logger.info("验证配置文件...")
    
    # 验证必需字段
    required_sections = ["model", "dataset", "lora", "training", "data_processing"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"配置文件缺少必需部分: {section}")
    
    # 验证模型配置
    model_config = config["model"]
    if "name" not in model_config:
        raise ValueError("模型配置缺少name字段")
    
    # 验证数据集配置
    dataset_config = config["dataset"]
    if "name" not in dataset_config:
        raise ValueError("数据集配置缺少name字段")
    
    # 验证LoRA配置
    lora_config = config["lora"]
    required_lora_fields = ["r", "lora_alpha", "lora_dropout", "bias", "task_type", "target_modules"]
    for field in required_lora_fields:
        if field not in lora_config:
            raise ValueError(f"LoRA配置缺少必需字段: {field}")
    
    # 验证训练配置
    training_config = config["training"]
    required_training_fields = [
        "output_dir", "num_train_epochs", "per_device_train_batch_size",
        "learning_rate", "logging_steps", "eval_steps", "save_steps"
    ]
    for field in required_training_fields:
        if field not in training_config:
            raise ValueError(f"训练配置缺少必需字段: {field}")
    
    # 验证数值范围
    if not isinstance(training_config["learning_rate"], (int, float)) or training_config["learning_rate"] <= 0:
        raise ValueError("学习率必须大于0")
    
    if not isinstance(training_config["num_train_epochs"], (int, float)) or training_config["num_train_epochs"] <= 0:
        raise ValueError("训练轮数必须大于0")
    
    if not isinstance(lora_config["r"], (int, float)) or lora_config["r"] <= 0:
        raise ValueError("LoRA rank必须大于0")
    
    logger.info("配置文件验证通过")


def get_device_info() -> Dict[str, Any]:
    """获取设备信息"""
    device_info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
        "device_name": torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU",
        "memory_allocated": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
        "memory_reserved": torch.cuda.memory_reserved() if torch.cuda.is_available() else 0,
    }
    
    if torch.cuda.is_available():
        device_info["total_memory"] = torch.cuda.get_device_properties(0).total_memory
    
    return device_info


def create_output_dir(output_dir: str) -> None:
    """创建输出目录"""
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"创建输出目录: {output_dir}")


def format_time(seconds: float) -> str:
    """格式化时间"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


def format_memory(bytes_value: int) -> str:
    """格式化内存大小"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f}{unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f}TB"


def check_dependencies() -> Dict[str, bool]:
    """检查依赖包"""
    dependencies = {
        "torch": False,
        "transformers": False,
        "peft": False,
        "datasets": False,
        "accelerate": False,
        "bitsandbytes": False,
    }
    
    try:
        import torch
        dependencies["torch"] = True
    except ImportError:
        pass
    
    try:
        import transformers
        dependencies["transformers"] = True
    except ImportError:
        pass
    
    try:
        import peft
        dependencies["peft"] = True
    except ImportError:
        pass
    
    try:
        import datasets
        dependencies["datasets"] = True
    except ImportError:
        pass
    
    try:
        import accelerate
        dependencies["accelerate"] = True
    except ImportError:
        pass
    
    try:
        import bitsandbytes
        dependencies["bitsandbytes"] = True
    except ImportError:
        pass
    
    return dependencies


def log_system_info() -> None:
    """记录系统信息"""
    import platform
    import psutil
    
    logger.info("=== 系统信息 ===")
    logger.info(f"操作系统: {platform.system()} {platform.release()}")
    logger.info(f"Python版本: {platform.python_version()}")
    logger.info(f"CPU核心数: {psutil.cpu_count()}")
    logger.info(f"内存总量: {format_memory(psutil.virtual_memory().total)}")
    
    # 检查依赖
    deps = check_dependencies()
    logger.info("依赖包状态:")
    for dep, available in deps.items():
        status = "✓" if available else "✗"
        logger.info(f"  {dep}: {status}")
    
    # 设备信息
    device_info = get_device_info()
    logger.info(f"CUDA可用: {device_info['cuda_available']}")
    if device_info['cuda_available']:
        logger.info(f"GPU数量: {device_info['device_count']}")
        logger.info(f"GPU名称: {device_info['device_name']}")
        logger.info(f"GPU内存: {format_memory(device_info['total_memory'])}")
    
    logger.info("================")
