#!/usr/bin/env python3
"""
训练脚本 - 优化版本
使用PEFT微调Gemma3-1b模型以支持工具调用
"""

import argparse
import logging
import sys
import os

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.utils import setup_logging, load_config, validate_config, get_device_info, log_system_info
from src.model_config import load_model_and_tokenizer
from src.data_processor import create_data_processor
from src.trainer import create_trainer


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="训练Gemma3-1b工具调用模型")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/training_config.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None,
        help="输出目录（覆盖配置文件中的设置）"
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
    
    # 记录系统信息
    log_system_info()
    
    logger.info("开始训练流程...")
    
    try:
        # 加载配置
        logger.info(f"加载配置文件: {args.config}")
        config = load_config(args.config)
        
        # 验证配置
        validate_config(config)
        
        # 覆盖输出目录（如果指定）
        if args.output_dir:
            config["training"]["output_dir"] = args.output_dir
        
        # 显示设备信息
        device_info = get_device_info()
        logger.info(f"设备信息: {device_info}")
        
        if not device_info["cuda_available"]:
            logger.warning("未检测到CUDA，训练将在CPU上进行（不推荐）")
        
        # 加载模型和分词器
        logger.info("加载模型和分词器...")
        model, tokenizer = load_model_and_tokenizer(config)
        
        # 创建数据处理器
        logger.info("创建数据处理器...")
        data_processor = create_data_processor(tokenizer, config)
        
        # 准备训练数据
        logger.info("准备训练数据...")
        dataset_name = config["dataset"]["name"]
        datasets = data_processor.prepare_training_data(dataset_name)
        
        train_dataset = datasets["train"]
        eval_dataset = datasets["eval"]
        
        logger.info(f"训练集大小: {len(train_dataset)}")
        logger.info(f"验证集大小: {len(eval_dataset)}")
        
        # 创建训练器
        logger.info("创建训练器...")
        trainer = create_trainer(model, tokenizer, config)
        
        # 开始训练
        logger.info("开始训练...")
        trainer.train(train_dataset, eval_dataset)
        
        # 最终评估
        logger.info("进行最终评估...")
        results = trainer.evaluate(eval_dataset)
        
        logger.info("训练完成!")
        logger.info(f"最终评估结果: {results}")
        
    except Exception as e:
        logger.error(f"训练过程中发生错误: {e}")
        raise


if __name__ == "__main__":
    main()
