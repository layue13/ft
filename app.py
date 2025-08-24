#!/usr/bin/env python3
"""
Gemma3-1b 工具调用微调 - Hugging Face Space版本
"""

import os
import sys
import logging
import torch
import gradio as gr
from pathlib import Path

# 添加src目录到Python路径
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils import setup_logging, load_config, validate_config, create_output_dir
from src.model_config import load_model_and_tokenizer
from src.data_processor import create_data_processor
from src.trainer import create_trainer


def setup_environment():
    """设置环境"""
    setup_logging("INFO")
    logger = logging.getLogger(__name__)
    
    # 检查CUDA
    logger.info(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU设备: {torch.cuda.get_device_name()}")
        logger.info(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    return logger


def start_training(config_path: str, output_dir: str = "./outputs", max_samples: int = 100):
    """开始训练"""
    logger = logging.getLogger(__name__)
    
    try:
        # 加载配置
        logger.info(f"加载配置文件: {config_path}")
        config = load_config(config_path)
        
        # 验证配置
        validate_config(config)
        
        # 设置输出目录
        config["training"]["output_dir"] = output_dir
        create_output_dir(output_dir)
        
        # 限制训练样本数
        config["dataset"]["max_samples"] = max_samples
        
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
        
        return f"✅ 训练完成!\n\n📊 评估结果:\n{results}\n\n💾 模型已保存到: {output_dir}"
        
    except Exception as e:
        error_msg = f"❌ 训练过程中发生错误: {e}"
        logger.error(error_msg)
        return error_msg


def create_interface():
    """创建Gradio界面"""
    
    with gr.Blocks(title="Gemma3-1b 工具调用微调", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🚀 Gemma3-1b 工具调用微调")
        gr.Markdown("使用PEFT微调Gemma3-1b模型以支持工具调用功能")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## ⚙️ 配置")
                
                with gr.Group():
                    config_path = gr.Textbox(
                        label="配置文件路径",
                        value="configs/space_config.yaml",
                        placeholder="配置文件路径",
                        info="训练配置文件路径 (推荐使用space_config.yaml)"
                    )
                    
                    output_dir = gr.Textbox(
                        label="输出目录",
                        value="./outputs",
                        placeholder="模型输出目录",
                        info="训练结果保存目录"
                    )
                    
                    max_samples = gr.Slider(
                        minimum=10,
                        maximum=1000,
                        value=100,
                        step=10,
                        label="最大训练样本数",
                        info="限制训练样本数量以节省时间"
                    )
                
                with gr.Group():
                    start_btn = gr.Button("🚀 开始训练", variant="primary", size="lg")
                    stop_btn = gr.Button("⏹️ 停止训练", variant="stop", size="lg")
            
            with gr.Column(scale=2):
                gr.Markdown("## 📊 训练状态")
                
                with gr.Group():
                    status_output = gr.Textbox(
                        label="训练日志",
                        lines=25,
                        max_lines=100,
                        interactive=False,
                        show_copy_button=True
                    )
                    
                    progress_bar = gr.Progress()
        
        # 绑定事件
        start_btn.click(
            fn=lambda: "开始训练...\n请等待模型加载...",
            outputs=status_output
        ).then(
            fn=start_training,
            inputs=[config_path, output_dir, max_samples],
            outputs=status_output,
            show_progress=True
        )
        
        stop_btn.click(
            fn=lambda: "训练已停止",
            outputs=status_output
        )
        
        gr.Markdown("""
        ## 📖 使用说明
        
        1. **准备阶段**: 确保配置文件存在，模型会自动下载
        2. **训练阶段**: 点击"开始训练"按钮，监控右侧日志
        3. **完成阶段**: 训练完成后模型会保存到指定目录
        
        ## 🖥️ 硬件信息
        
        - **GPU**: 当前Space配置的GPU
        - **内存**: 自动分配
        - **存储**: 50GB可用空间
        
        ## 💡 优化建议
        
        - 调整"最大训练样本数"以控制训练时间
        - 使用较小的batch size以节省显存
        - 监控训练日志以了解进度
        
        ## 🆘 常见问题
        
        - **模型下载失败**: 检查网络连接和HF Token
        - **显存不足**: 减少batch size或训练样本数
        - **训练中断**: 检查Space日志获取详细错误信息
        
        ## 🔗 相关链接
        
        - [项目文档](https://github.com/your-repo)
        - [数据集信息](https://huggingface.co/datasets/shawhin/tool-use-finetuning)
        - [Gemma3模型](https://huggingface.co/google/gemma-3-1b-it)
        """)
    
    return demo


if __name__ == "__main__":
    # 设置环境
    logger = setup_environment()
    
    # 创建界面
    demo = create_interface()
    
    # 启动应用
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )
