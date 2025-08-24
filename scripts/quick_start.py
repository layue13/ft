#!/usr/bin/env python3
"""
快速启动脚本
提供多种微调选项的快速启动
"""

import argparse
import logging
import sys
import os
from pathlib import Path

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.utils import setup_logging


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Gemma3-1b 工具调用微调 - 快速启动")
    parser.add_argument(
        "--mode", 
        type=str, 
        default="local",
        choices=["local", "hf-jobs", "hf-spaces"],
        help="运行模式"
    )
    parser.add_argument(
        "--flavor", 
        type=str, 
        default="a10g-small",
        choices=["t4-small", "t4-medium", "a10g-small", "a10g-large", "a100-large"],
        help="硬件配置 (仅HF Jobs模式)"
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
    
    logger.info("🚀 Gemma3-1b 工具调用微调 - 快速启动")
    
    if args.mode == "local":
        logger.info("本地训练模式")
        logger.info("请确保您有足够的GPU资源")
        logger.info("运行命令: python scripts/train.py")
        
        # 检查环境
        try:
            import torch
            if torch.cuda.is_available():
                logger.info(f"✅ 检测到GPU: {torch.cuda.get_device_name()}")
                logger.info(f"✅ GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            else:
                logger.warning("⚠️ 未检测到CUDA，训练将在CPU上进行（不推荐）")
        except ImportError:
            logger.error("❌ 未安装PyTorch")
            return
        
        # 运行本地训练
        os.system("python scripts/train.py")
        
    elif args.mode == "hf-jobs":
        logger.info("Hugging Face Jobs模式")
        logger.info(f"硬件配置: {args.flavor}")
        
        # 检查HF Token
        if not os.environ.get("HF_TOKEN"):
            logger.error("❌ 请设置HF_TOKEN环境变量")
            logger.info("运行: export HF_TOKEN='your_token'")
            return
        
        logger.info("✅ HF Token已设置")
        
        # 运行HF Jobs训练
        cmd = f"python scripts/train_hf_jobs.py --flavor {args.flavor}"
        logger.info(f"运行命令: {cmd}")
        os.system(cmd)
        
    elif args.mode == "hf-spaces":
        logger.info("Hugging Face Spaces模式")
        logger.info("请按照以下步骤操作:")
        logger.info("1. 将此项目推送到Hugging Face Hub")
        logger.info("2. 创建新的Space，选择Gradio模板")
        logger.info("3. 升级到GPU硬件")
        logger.info("4. 通过Web界面启动训练")
        
        # 检查是否在Space环境中
        if os.environ.get("SPACE_ID"):
            logger.info("✅ 检测到Space环境")
            logger.info("启动Space应用...")
            os.system("python app.py")
        else:
            logger.info("当前不在Space环境中")
            logger.info("请部署到Hugging Face Spaces后重试")


if __name__ == "__main__":
    main()
