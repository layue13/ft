#!/usr/bin/env python3
"""
使用Hugging Face Jobs进行微调的脚本
"""

import argparse
import logging
import sys
import os
from pathlib import Path

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.utils import setup_logging, load_config, validate_config


def create_uv_script(config_path: str, output_dir: str = "./hf_outputs") -> str:
    """创建UV脚本用于Hugging Face Jobs"""
    
    script_content = f'''#!/usr/bin/env python3
"""
Gemma3-1b 工具调用微调 - Hugging Face Jobs版本
"""

# uv: dependencies
# torch>=2.0.0
# transformers>=4.40.0
# peft>=0.7.0
# datasets>=2.14.0
# accelerate>=0.20.0
# tqdm>=4.65.0
# wandb>=0.15.0
# numpy>=1.24.0
# scipy>=1.10.0
# pyyaml>=6.0
# sentencepiece>=0.1.99

import os
import sys
import yaml
import logging
import torch
from pathlib import Path

# 添加项目代码到路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.utils import setup_logging, load_config, validate_config, create_output_dir
from src.model_config import load_model_and_tokenizer
from src.data_processor import create_data_processor
from src.trainer import create_trainer


def main():
    """主函数"""
    setup_logging("INFO")
    logger = logging.getLogger(__name__)
    
    logger.info("开始Hugging Face Jobs微调...")
    
    # 显示环境信息
    logger.info(f"CUDA可用: {{torch.cuda.is_available()}}")
    if torch.cuda.is_available():
        logger.info(f"GPU设备: {{torch.cuda.get_device_name()}}")
        logger.info(f"GPU内存: {{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}} GB")
    
    try:
        # 加载配置
        config_path = "{config_path}"
        logger.info(f"加载配置文件: {{config_path}}")
        config = load_config(config_path)
        
        # 验证配置
        validate_config(config)
        
        # 设置输出目录
        config["training"]["output_dir"] = "{output_dir}"
        create_output_dir(config["training"]["output_dir"])
        
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
        
        logger.info(f"训练集大小: {{len(train_dataset)}}")
        logger.info(f"验证集大小: {{len(eval_dataset)}}")
        
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
        logger.info(f"最终评估结果: {{results}}")
        
        # 保存到Hugging Face Hub (可选)
        if os.environ.get("HF_TOKEN"):
            logger.info("保存模型到Hugging Face Hub...")
            model.push_to_hub("gemma3-tool-finetuned", use_auth_token=os.environ["HF_TOKEN"])
            tokenizer.push_to_hub("gemma3-tool-finetuned", use_auth_token=os.environ["HF_TOKEN"])
        
    except Exception as e:
        logger.error(f"训练过程中发生错误: {{e}}")
        raise


if __name__ == "__main__":
    main()
'''
    
    # 保存脚本
    script_path = "hf_training_script.py"
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    return script_path


def run_hf_job(script_path: str, config_path: str, flavor: str = "a10g-small"):
    """运行Hugging Face Job"""
    
    try:
        from huggingface_hub import run_uv_job
        
        # 设置环境变量
        env_vars = {
            "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
            "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", ""),
        }
        
        # 运行Job
        logger.info(f"启动Hugging Face Job，使用硬件: {{flavor}}")
        job = run_uv_job(
            script_path,
            flavor=flavor,
            env=env_vars,
            secrets={"HF_TOKEN": os.environ.get("HF_TOKEN", "")}
        )
        
        logger.info(f"Job已启动: {{job.url}}")
        logger.info(f"Job ID: {{job.id}}")
        
        return job
        
    except ImportError:
        logger.error("请安装huggingface_hub: pip install huggingface_hub")
        return None
    except Exception as e:
        logger.error(f"启动Job失败: {{e}}")
        return None


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="使用Hugging Face Jobs进行微调")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/training_config.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--flavor", 
        type=str, 
        default="a10g-small",
        choices=["t4-small", "t4-medium", "a10g-small", "a10g-large", "a100-large"],
        help="硬件配置"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./hf_outputs",
        help="输出目录"
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
    
    logger.info("准备Hugging Face Jobs微调...")
    
    try:
        # 验证配置
        config = load_config(args.config)
        validate_config(config)
        
        # 创建UV脚本
        logger.info("创建UV脚本...")
        script_path = create_uv_script(args.config, args.output_dir)
        
        # 运行Job
        job = run_hf_job(script_path, args.config, args.flavor)
        
        if job:
            logger.info("Job启动成功!")
            logger.info(f"查看进度: {{job.url}}")
            logger.info("注意: 这是一个付费服务，请监控您的使用情况")
        else:
            logger.error("Job启动失败")
            
    except Exception as e:
        logger.error(f"准备过程中发生错误: {{e}}")
        raise


if __name__ == "__main__":
    main()
