#!/usr/bin/env python3
"""
Gemma-3-1b Tool Use 微调脚本
基于第一性原理：模型 + 数据 + 训练循环
目标：让Gemma-3-1b支持工具调用
"""

import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, 
    TrainingArguments, Trainer
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

def main():
    print("🚀 开始Gemma-3-1b Tool Use微调...")
    
    # 1. 模型和分词器 - Gemma-3-1b-it
    model_name = "google/gemma-3-1b-it"  # 使用正确的Gemma-3-1b模型
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="right")  # Gemma使用right padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 2. LoRA配置 - 针对Gemma模型
    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.1,  # 增大rank以提高工具调用能力
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]  # Gemma特定模块
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 3. Tool Use数据 - shawhin/tool-use-finetuning数据集
    dataset = load_dataset("shawhin/tool-use-finetuning", split="train[:200]")  # 增加样本量
    
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
    
    dataset = dataset.map(format_tool_use_data)
    
    def tokenize(examples):
        return tokenizer(
            examples["text"], 
            truncation=True, 
            padding="max_length",
            max_length=512,  # 工具调用需要更长序列
            return_tensors="pt"
        )
    
    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
    tokenized = tokenized.add_column("labels", tokenized["input_ids"])
    
    # 4. 训练参数 - Tool Use优化
    training_args = TrainingArguments(
        output_dir="./gemma3-tool-use",
        num_train_epochs=2,  # 工具调用需要更多训练
        per_device_train_batch_size=1,  # Gemma模型较大，减小batch size
        gradient_accumulation_steps=4,  # 通过梯度累积增加有效batch size
        learning_rate=2e-5,  # 降低学习率，避免破坏预训练知识
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        push_to_hub=True,
        hub_model_id="gemma3-1b-tool-use",  # 指定Hub模型名
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        bf16=True,  # 使用bf16提高效率
        gradient_checkpointing=True,  # 节省显存
    )
    
    # 5. 训练器 - Tool Use专用
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        tokenizer=tokenizer
    )
    
    # 6. 开始训练
    print("✨ 开始Tool Use微调...")
    print(f"📊 训练样本: {len(tokenized)}")
    print(f"🎯 目标: 让Gemma-3-1b学会工具调用")
    
    trainer.train()
    
    # 7. 保存最终模型
    trainer.save_model()
    tokenizer.save_pretrained("./gemma3-tool-use")
    
    print("🎉 Gemma-3-1b Tool Use微调完成！")
    print("📤 模型已推送到Hugging Face Hub")
    print("🛠 现在可以使用工具调用功能了！")

if __name__ == "__main__":
    main()