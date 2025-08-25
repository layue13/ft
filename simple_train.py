#!/usr/bin/env python3
"""
最简化的HF Jobs训练脚本
基于第一性原理：模型 + 数据 + 训练循环
"""

import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, 
    TrainingArguments, Trainer
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

def main():
    print("🚀 开始极简训练...")
    
    # 1. 模型和分词器 - 使用公开可用的小模型
    model_name = "microsoft/DialoGPT-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # 2. LoRA配置 - 最小设置
    lora_config = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.1,
        task_type=TaskType.CAUSAL_LM,
        target_modules=["c_attn", "c_proj"]
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 3. 数据 - 使用小数据集快速测试
    dataset = load_dataset("shawhin/tool-use-finetuning", split="train[:100]")
    
    def format_data(example):
        if "trace" not in example:
            return {"text": "user: hello\nassistant: hi"}
        
        text = ""
        for msg in example["trace"][:3]:  # 只取前3个消息
            role = msg.get("role", "user")
            content = str(msg.get("content", ""))[:100]  # 限制长度
            text += f"{role}: {content}\n"
        return {"text": text}
    
    dataset = dataset.map(format_data)
    
    def tokenize(examples):
        return tokenizer(
            examples["text"], 
            truncation=True, 
            padding="max_length",
            max_length=256,  # 更短的序列
            return_tensors="pt"
        )
    
    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
    tokenized = tokenized.add_column("labels", tokenized["input_ids"])
    
    # 4. 训练参数 - 最小化设置
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,  # 只训练1个epoch
        per_device_train_batch_size=2,
        learning_rate=1e-4,
        logging_steps=5,
        save_strategy="no",  # 不保存中间检查点
        push_to_hub=True,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )
    
    # 5. 训练器 - 无验证集，快速训练
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        tokenizer=tokenizer
    )
    
    # 6. 开始训练
    print("✨ 训练中...")
    trainer.train()
    
    print("🎉 训练完成！模型已推送到Hub")

if __name__ == "__main__":
    main()