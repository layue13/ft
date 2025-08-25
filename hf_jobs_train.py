#!/usr/bin/env python3
"""
HF Jobs云端训练脚本
优化用于Hugging Face Jobs平台的模型微调
"""

import os
import sys
import logging
from pathlib import Path

# 设置环境变量
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"  # 在云端训练中禁用wandb

# 导入必要的库
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_and_prepare_model():
    """加载和准备模型"""
    model_name = "microsoft/DialoGPT-small"  # 使用公开可用的模型
    
    logger.info(f"加载模型: {model_name}")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="left",
        use_fast=False
    )
    
    # 设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 创建LoRA配置
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["c_attn", "c_proj"],  # DialoGPT特定模块
        inference_mode=False,
    )
    
    # 应用LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer


def prepare_dataset(tokenizer, max_samples=500):
    """准备训练数据集"""
    logger.info("准备数据集...")
    
    try:
        # 加载数据集
        dataset = load_dataset("shawhin/tool-use-finetuning", split="train")
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
            
        logger.info(f"数据集大小: {len(dataset)}")
        
        def format_conversation(example):
            """格式化对话"""
            if "trace" not in example:
                return {"text": ""}
                
            conversation = ""
            for message in example["trace"]:
                role = message.get("role", "")
                content = message.get("content", "")
                
                if isinstance(content, dict):
                    # 处理工具调用
                    content = f"工具调用: {content.get('name', 'unknown')}"
                
                conversation += f"{role}: {content}\n"
            
            return {"text": conversation}
        
        # 格式化数据
        dataset = dataset.map(format_conversation)
        
        def tokenize_function(examples):
            """分词函数"""
            texts = examples["text"]
            tokenized = tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=512,
                return_tensors="pt"
            )
            
            # 设置labels
            tokenized["labels"] = tokenized["input_ids"].clone()
            
            return tokenized
        
        # 应用分词
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # 分割数据集
        train_size = int(0.8 * len(tokenized_dataset))
        train_dataset = tokenized_dataset.select(range(train_size))
        eval_dataset = tokenized_dataset.select(range(train_size, len(tokenized_dataset)))
        
        logger.info(f"训练集: {len(train_dataset)}, 验证集: {len(eval_dataset)}")
        
        return train_dataset, eval_dataset
        
    except Exception as e:
        logger.error(f"数据集准备失败: {e}")
        # 创建虚拟数据集用于测试
        dummy_data = {
            "text": [
                "user: 你好\nassistant: 你好！有什么可以帮助你的吗？",
                "user: 今天天气怎么样？\nassistant: 我无法获取实时天气信息。",
            ] * 50
        }
        
        from datasets import Dataset
        dataset = Dataset.from_dict(dummy_data)
        
        def tokenize_function(examples):
            tokenized = tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=512,
                return_tensors="pt"
            )
            tokenized["labels"] = tokenized["input_ids"].clone()
            return tokenized
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        train_dataset = tokenized_dataset.select(range(80))
        eval_dataset = tokenized_dataset.select(range(80, 100))
        
        return train_dataset, eval_dataset


def create_training_arguments():
    """创建训练参数"""
    return TrainingArguments(
        output_dir="./results",
        num_train_epochs=2,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_steps=50,
        logging_steps=10,
        eval_steps=100,
        save_steps=100,
        save_total_limit=2,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=False,  # 禁用fp16避免版本兼容性问题
        dataloader_pin_memory=False,  # 禁用以提高兼容性
        push_to_hub=True,  # 自动推送到HF Hub
        report_to="none",  # 禁用wandb
        remove_unused_columns=False,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        gradient_checkpointing=False,  # 禁用以避免accelerate版本问题
        dataloader_num_workers=0,
    )


def main():
    """主训练函数"""
    logger.info("开始HF Jobs云端训练...")
    
    try:
        # 检查库版本兼容性
        import transformers
        import accelerate
        logger.info(f"Transformers版本: {transformers.__version__}")
        logger.info(f"Accelerate版本: {accelerate.__version__}")
        # 检查GPU可用性
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"使用设备: {device}")
        
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name()}")
            logger.info(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        
        # 加载模型和分词器
        model, tokenizer = load_and_prepare_model()
        
        # 准备数据集
        train_dataset, eval_dataset = prepare_dataset(tokenizer)
        
        # 创建训练参数
        training_args = create_training_arguments()
        
        # 创建训练器
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
        )
        
        # 开始训练
        logger.info("开始训练...")
        trainer.train()
        
        # 保存模型
        logger.info("保存模型...")
        trainer.save_model()
        
        # 最终评估
        logger.info("进行最终评估...")
        results = trainer.evaluate()
        logger.info(f"最终评估结果: {results}")
        
        logger.info("训练完成！")
        
    except Exception as e:
        logger.error(f"训练过程中发生错误: {e}")
        raise


if __name__ == "__main__":
    main()