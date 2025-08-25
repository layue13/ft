#!/usr/bin/env python3
# /// script
# dependencies = [
#     "torch>=2.2.0",
#     "transformers>=4.40.0", 
#     "peft>=0.7.1",
#     "datasets>=2.14.0",
#     "accelerate>=0.26.0",
#     "huggingface-hub[cli]>=0.20.0"
# ]
# ///
"""
Gemma-3-1b Tool Use 微调脚本
第一性原理: 模型 + 数据 + 训练 = 工具调用能力
"""

import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import load_dataset
from huggingface_hub import login

# 常量配置
MODEL_NAME = "google/gemma-3-1b-it"
DATASET = "shawhin/tool-use-finetuning"
DATASET_SIZE = 200
OUTPUT_DIR = "./gemma3-tool-use"
MAX_LENGTH = 512
BATCH_SIZE = 1
GRAD_ACCUMULATION = 4
LEARNING_RATE = 2e-5
EPOCHS = 2
HF_REPO_ID = "gemma3-tool-use"  # 修改为你的HF用户名

def setup_device():
    """设备和精度配置"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16_supported else torch.float32
    print(f"Device: {device}, dtype: {dtype}")
    return device, bf16_supported, dtype

def load_model_and_tokenizer(model_name, dtype, device):
    """加载模型和分词器"""
    print(f"Loading: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="right")
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        attn_implementation="eager"
    )
    return model, tokenizer

def apply_lora(model):
    """应用LoRA配置"""
    # 先冻结所有参数
    for param in model.parameters():
        param.requires_grad = False
    
    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.1,
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, lora_config)
    
    # 确保LoRA参数可训练
    for name, param in model.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True
    
    model.print_trainable_parameters()
    return model

def format_tool_data(example):
    """将数据格式化为Gemma对话格式"""
    if not example.get("tool_needed") or "trace" not in example:
        return {"text": "<bos><start_of_turn>user\nhello<end_of_turn>\n<start_of_turn>model\nHello! How can I help?<end_of_turn><eos>"}
    
    conversation = "<bos>"
    for msg in example["trace"]:
        if msg.get("role") == "user":
            conversation += f"<start_of_turn>user\n{msg.get('content', '')}<end_of_turn>\n"
    
    if example.get("tool_name"):
        tool_call = f'<tool_call>\n{{\n "tool_name": "{example["tool_name"]}",\n "args": {{}}\n}}\n</tool_call>'
        conversation += f"<start_of_turn>model\n{tool_call}<end_of_turn>"
    
    return {"text": conversation + "<eos>"}

def prepare_dataset(dataset_name, size, tokenizer, max_length):
    """准备训练数据"""
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split=f"train[:{size}]")
    dataset = dataset.map(format_tool_data)
    
    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length)
    
    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
    tokenized = tokenized.add_column("labels", tokenized["input_ids"])
    print(f"Dataset ready: {len(tokenized)} samples")
    return tokenized

def create_trainer(model, tokenizer, train_dataset, output_dir, bf16_supported, device):
    """创建训练器"""
    hf_token = os.getenv("HF_TOKEN")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        bf16=bf16_supported,
        gradient_checkpointing=False,
        dataloader_pin_memory=device != "cpu",
        # 禁用自动Hub上传，避免权限问题
        push_to_hub=False
    )
    
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

def merge_and_save_model(lora_model_path, output_path):
    """合并LoRA权重到基础模型"""
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base_model, lora_model_path)
    merged_model = model.merge_and_unload()
    
    merged_model.save_pretrained(output_path)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.save_pretrained(output_path)

def upload_to_hub(model_path, repo_id):
    """上传模型到Hugging Face Hub"""
    if os.getenv("HF_TOKEN"):
        login(token=os.getenv("HF_TOKEN"))
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    model.push_to_hub(repo_id, private=False)
    tokenizer.push_to_hub(repo_id, private=False)

def main():
    print("🚀 Gemma-3-1b Tool Use Fine-tuning")
    
    # 设备配置
    device, bf16_supported, dtype = setup_device()
    
    # 模型加载
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME, dtype, device)
    model = apply_lora(model)
    
    # 数据准备
    train_dataset = prepare_dataset(DATASET, DATASET_SIZE, tokenizer, MAX_LENGTH)
    
    # 训练配置
    trainer = create_trainer(model, tokenizer, train_dataset, OUTPUT_DIR, bf16_supported, device)
    
    # 开始训练
    print(f"Training: {len(train_dataset)} samples, {EPOCHS} epochs")
    train_result = trainer.train()
    
    # 保存模型（标准做法）
    trainer.save_model()  # 保存模型和分词器
    print(f"✅ LoRA model saved to: {OUTPUT_DIR}")
    
    # 保存训练指标
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    # 合并LoRA权重
    print("🔗 Merging LoRA weights...")
    merged_dir = f"{OUTPUT_DIR}-merged"
    merge_and_save_model(OUTPUT_DIR, merged_dir)
    print(f"✅ Merged model saved to: {merged_dir}")
    
    # 可选的Hub上传（使用标准方法）
    if os.getenv("HF_TOKEN"):
        print("📤 Trying to upload to Hugging Face Hub...")
        try:
            # 加载合并后的模型并尝试上传
            from transformers import AutoModelForCausalLM, AutoTokenizer
            model = AutoModelForCausalLM.from_pretrained(merged_dir)
            tokenizer = AutoTokenizer.from_pretrained(merged_dir)
            
            model.push_to_hub(HF_REPO_ID, private=False)
            tokenizer.push_to_hub(HF_REPO_ID, private=False)
            print(f"✅ Model uploaded to: https://huggingface.co/{HF_REPO_ID}")
        except Exception as e:
            print(f"⚠️ Upload failed: {e}")
            print("💾 Model saved locally for manual upload")
    else:
        print("💾 Models saved locally. Set HF_TOKEN to upload to Hub")

if __name__ == "__main__":
    main()