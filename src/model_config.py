"""
模型配置模块
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, TaskType
from typing import Dict, Any, Tuple, Optional


def create_quantization_config(config: Dict[str, Any]) -> Optional[BitsAndBytesConfig]:
    """创建量化配置"""
    import platform
    
    # 在macOS上跳过量化
    if platform.system() == "Darwin":
        return None
    
    try:
        import bitsandbytes
        model_config = config["model"]
        
        return BitsAndBytesConfig(
            load_in_4bit=model_config.get("use_4bit", True),
            bnb_4bit_use_double_quant=model_config.get("use_nested_quant", True),
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=getattr(torch, model_config.get("bnb_4bit_compute_dtype", "bfloat16"))
        )
    except ImportError:
        # 如果没有安装bitsandbytes，返回None
        return None


def create_lora_config(config: Dict[str, Any]) -> LoraConfig:
    """创建LoRA配置"""
    lora_config = config["lora"]
    
    return LoraConfig(
        r=lora_config["r"],
        lora_alpha=lora_config["lora_alpha"],
        lora_dropout=lora_config["lora_dropout"],
        bias=lora_config["bias"],
        task_type=getattr(TaskType, lora_config["task_type"]),
        target_modules=lora_config["target_modules"],
        inference_mode=False,
    )


def create_training_arguments(config: Dict[str, Any]) -> TrainingArguments:
    """创建训练参数"""
    training_config = config["training"]
    
    return TrainingArguments(
        output_dir=training_config["output_dir"],
        num_train_epochs=training_config["num_train_epochs"],
        per_device_train_batch_size=training_config["per_device_train_batch_size"],
        per_device_eval_batch_size=training_config["per_device_eval_batch_size"],
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        learning_rate=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
        warmup_steps=training_config["warmup_steps"],
        logging_steps=training_config["logging_steps"],
        eval_steps=training_config["eval_steps"],
        save_steps=training_config["save_steps"],
        save_total_limit=training_config["save_total_limit"],
        evaluation_strategy=training_config["evaluation_strategy"],
        save_strategy=training_config["save_strategy"],
        load_best_model_at_end=training_config["load_best_model_at_end"],
        metric_for_best_model=training_config["metric_for_best_model"],
        greater_is_better=training_config["greater_is_better"],
        fp16=training_config["fp16"],
        dataloader_pin_memory=training_config["dataloader_pin_memory"],
        remove_unused_columns=training_config["remove_unused_columns"],
        push_to_hub=training_config["push_to_hub"],
        report_to=training_config["report_to"],
        # 额外的优化设置
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        dataloader_num_workers=4,
        group_by_length=True,
    )


def load_model_and_tokenizer(
    config: Dict[str, Any]
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """加载模型和分词器"""
    model_name = config["model"]["name"]
    
    # 获取镜像站配置
    mirror_name = config.get("mirror", {}).get("name")
    if mirror_name:
        from .mirror_utils import get_mirror_selector
        selector = get_mirror_selector()
        model_url = selector.get_model_url(model_name, mirror_name)
        logger.info(f"使用镜像站 {mirror_name} 加载模型: {model_url}")
    else:
        model_url = model_name
        logger.info(f"使用官方源加载模型: {model_url}")

    # 创建量化配置
    quantization_config = create_quantization_config(config)

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_url,
        trust_remote_code=True,
        padding_side="left",  # Gemma3使用left padding
        use_fast=False,
    )
    
    # 设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    model_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "sdpa",
    }
    
    # 只有在量化配置可用时才使用
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    
    # 对于Gemma3-1b-it，使用Gemma3ForConditionalGeneration
    if "gemma-3" in model_name:
        from transformers import Gemma3ForConditionalGeneration
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_url,
            **model_kwargs
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_url,
            **model_kwargs
        )
    
    # 创建LoRA配置
    lora_config = create_lora_config(config)
    
    # 应用LoRA
    model = get_peft_model(model, lora_config)
    
    # 打印可训练参数
    model.print_trainable_parameters()
    
    return model, tokenizer


def prepare_model_for_training(model: AutoModelForCausalLM) -> AutoModelForCausalLM:
    """准备模型进行训练"""
    # 启用梯度检查点以节省内存
    model.gradient_checkpointing_enable()
    
    # 启用模型并行
    model.enable_input_require_grads()
    
    return model
