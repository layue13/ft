#!/usr/bin/env python3
"""
Gemma-3-1b Tool Use 微调脚本 - HF Jobs版本
基于第一性原理：模型 + 数据 + 训练循环
目标：让Gemma-3-1b支持工具调用
"""

import os
import subprocess
import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, 
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset, Dataset
from huggingface_hub import login
import warnings

# 忽略警告
warnings.filterwarnings("ignore", category=FutureWarning, module="datasets")
warnings.filterwarnings("ignore", category=SyntaxWarning, module="peft")
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.data.dataloader")

def setup_environment():
    """设置环境"""
    print("🔧 设置环境...")
    
    # 使用uv安装依赖
    print("📦 使用uv安装依赖...")
    subprocess.check_call(["uv", "sync"])

def main():
    print("🚀 开始Gemma-3-1b Tool Use微调 (HF Jobs版本)...")
    
    # 设置环境
    setup_environment()
    
    # 登录Hugging Face
    if os.getenv("HF_TOKEN"):
        login(token=os.getenv("HF_TOKEN"))
        print("✅ 已登录Hugging Face")
    
    # 检查设备支持
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 使用设备: {device}")
    
    # 检查bf16支持
    bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    print(f"🔧 bf16支持: {bf16_supported}")
    
    # 1. 模型和分词器 - Gemma-3-1b
    model_name = "google/gemma-3-1b-it"  # 使用Gemma-3-1b模型
    print(f"📦 加载模型: {model_name}")
    
    # 加载tokenizer，使用安全配置
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        padding_side="right",
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型，使用eager注意力机制
    torch_dtype = torch.bfloat16 if bf16_supported else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
        attn_implementation="eager"  # 使用eager注意力机制
    )
    
    # 2. LoRA配置 - 针对Gemma模型
    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.1,
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 3. Tool Use数据 - 使用真实的工具调用数据集
    print("📊 准备训练数据...")
    
    # 加载真实的工具调用数据集
    dataset = load_dataset("shawhin/tool-use-finetuning", split="train[:200]")
    print(f"📦 加载数据集: {len(dataset)} 个样本")
    
    def format_tool_use_data(example):
        """格式化工具调用数据为Gemma对话格式"""
        if "trace" not in example or not example.get("tool_needed"):
            return {"text": "<bos><start_of_turn>user\nhello<end_of_turn>\n<start_of_turn>model\nHello! How can I help you?<end_of_turn><eos>"}
        
        conversation = "<bos>"
        
        # 处理对话历史
        for msg in example["trace"]:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            if role == "user":
                conversation += f"<start_of_turn>user\n{content}<end_of_turn>\n"
        
        # 添加工具调用响应
        if example.get("tool_needed") and example.get("tool_name"):
            tool_call = f'<tool_call>\n{{\n "tool_name": "{example["tool_name"]}",\n "args": {{}}\n}}\n</tool_call>'
            conversation += f"<start_of_turn>model\n{tool_call}<end_of_turn>"
        
        conversation += "<eos>"
        return {"text": conversation}
    
    # 格式化数据集
    dataset = dataset.map(format_tool_use_data)
    print(f"✅ 数据格式化完成")
    
    def tokenize(examples):
        return tokenizer(
            examples["text"], 
            truncation=True, 
            padding="max_length",
            max_length=512,  # Gemma需要更长序列
            return_tensors="pt"
        )
    
    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
    tokenized = tokenized.add_column("labels", tokenized["input_ids"])
    
    # 4. 数据整理器 - 处理批处理
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # 使用因果语言建模
    )
    
    # 5. 训练参数 - HF Jobs优化
    training_args = TrainingArguments(
        output_dir="./gemma3-tool-use",
        num_train_epochs=3,  # 增加训练轮数
        per_device_train_batch_size=2,  # 增加batch size
        gradient_accumulation_steps=8,  # 增加梯度累积
        learning_rate=2e-5,  # 较低学习率避免破坏预训练知识
        warmup_ratio=0.1,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=3,
        push_to_hub=True,  # 推送到Hub
        hub_model_id="gemma3-1b-tool-use",  # 指定Hub模型名
        report_to="wandb",  # 使用wandb记录
        remove_unused_columns=False,
        dataloader_num_workers=2,  # 增加数据加载器工作进程
        bf16=bf16_supported,
        gradient_checkpointing=True,  # 启用梯度检查点
        dataloader_pin_memory=True,  # 启用pin_memory
    )
    
    # 6. 训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )
    
    # 7. 开始训练
    print("✨ 开始Gemma-3-1b Tool Use微调...")
    print(f"📊 训练样本: {len(tokenized)}")
    print(f"🎯 目标: 让Gemma-3-1b学会工具调用")
    print(f"⚙️ 训练配置: batch_size={training_args.per_device_train_batch_size}, "
          f"gradient_accumulation={training_args.gradient_accumulation_steps}, "
          f"learning_rate={training_args.learning_rate}")
    
    trainer.train()
    
    # 8. 保存最终模型
    trainer.save_model()
    tokenizer.save_pretrained("./gemma3-tool-use")
    
    # 9. 推送到Hub
    if os.getenv("HF_TOKEN"):
        trainer.push_to_hub()
        print("📤 模型已推送到Hugging Face Hub")
    
    print("🎉 Gemma-3-1b Tool Use微调完成！")
    print("💾 模型已保存到 ./gemma3-tool-use")
    print("🛠 现在可以使用工具调用功能了！")

if __name__ == "__main__":
    main()
