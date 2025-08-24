"""
训练器模块
"""

import logging
import os
from typing import Dict, Any, Optional
from transformers import Trainer, TrainingArguments
from peft import PeftModel
from .utils import get_device_info, create_output_dir
from .model_config import prepare_model_for_training


logger = logging.getLogger(__name__)


class ToolUseTrainer:
    """工具调用训练器"""
    
    def __init__(self, model, tokenizer, config: Dict[str, Any]):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.trainer = None
        
        # 准备模型进行训练
        self.model = prepare_model_for_training(self.model)
        
        # 创建输出目录
        create_output_dir(config["training"]["output_dir"])
        
        # 记录设备信息
        device_info = get_device_info()
        logger.info(f"设备信息: {device_info}")
    
    def create_trainer(self, train_dataset, eval_dataset) -> Trainer:
        """创建训练器"""
        from .model_config import create_training_arguments
        
        training_args = create_training_arguments(self.config)
        
        # 添加详细的日志回调
        from transformers import TrainingArguments
        training_args.logging_strategy = "steps"
        training_args.logging_first_step = True
        training_args.logging_steps = 1  # 每步都打印日志
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=self._data_collator,
        )
        
        return self.trainer
    
    def _data_collator(self, features):
        """数据整理器"""
        import torch
        
        batch_size = len(features)
        max_length = max(len(f["input_ids"]) for f in features)
        
        batch = {
            "input_ids": torch.zeros((batch_size, max_length), dtype=torch.long),
            "attention_mask": torch.zeros((batch_size, max_length), dtype=torch.long),
            "labels": torch.full((batch_size, max_length), -100, dtype=torch.long),
        }
        
        for i, feature in enumerate(features):
            input_ids = feature["input_ids"]
            attention_mask = feature["attention_mask"]
            labels = feature["labels"]
            
            length = len(input_ids)
            batch["input_ids"][i, :length] = torch.tensor(input_ids)
            batch["attention_mask"][i, :length] = torch.tensor(attention_mask)
            batch["labels"][i, :length] = torch.tensor(labels)
        
        return batch
    
    def train(self, train_dataset, eval_dataset) -> None:
        """开始训练"""
        logger.info("开始训练...")
        logger.info(f"训练集大小: {len(train_dataset)}")
        logger.info(f"验证集大小: {len(eval_dataset)}")
        
        # 创建训练器
        trainer = self.create_trainer(train_dataset, eval_dataset)
        
        # 打印训练配置
        logger.info("=== 训练配置 ===")
        logger.info(f"学习率: {trainer.args.learning_rate}")
        logger.info(f"Batch Size: {trainer.args.per_device_train_batch_size}")
        logger.info(f"梯度累积步数: {trainer.args.gradient_accumulation_steps}")
        logger.info(f"训练轮数: {trainer.args.num_train_epochs}")
        logger.info(f"总训练步数: {trainer.args.num_train_epochs * len(train_dataset) // (trainer.args.per_device_train_batch_size * trainer.args.gradient_accumulation_steps)}")
        logger.info(f"评估步数: {trainer.args.eval_steps}")
        logger.info(f"保存步数: {trainer.args.save_steps}")
        logger.info(f"日志步数: {trainer.args.logging_steps}")
        logger.info("==================")
        
        # 开始训练
        logger.info("开始模型训练...")
        train_result = trainer.train()
        
        # 打印训练结果
        logger.info("=== 训练结果 ===")
        logger.info(f"训练损失: {train_result.training_loss}")
        logger.info(f"训练步数: {train_result.global_step}")
        logger.info(f"训练时间: {train_result.metrics.get('train_runtime', 'N/A')} 秒")
        logger.info("================")
        
        # 保存最终模型
        self.save_model()
        
        logger.info("训练完成!")
    
    def save_model(self, output_dir: Optional[str] = None) -> None:
        """保存模型"""
        if output_dir is None:
            output_dir = self.config["training"]["output_dir"]
        
        logger.info(f"保存模型到: {output_dir}")
        
        # 保存PEFT适配器
        self.model.save_pretrained(output_dir)
        
        # 保存分词器
        self.tokenizer.save_pretrained(output_dir)
        
        # 保存配置
        from .utils import save_config
        save_config(self.config, os.path.join(output_dir, "training_config.yaml"))
        
        logger.info("模型保存完成!")
    
    def evaluate(self, eval_dataset) -> Dict[str, float]:
        """评估模型"""
        if self.trainer is None:
            raise ValueError("训练器未初始化，请先调用train()方法")
        
        logger.info("开始评估...")
        logger.info(f"评估数据集大小: {len(eval_dataset)}")
        
        results = self.trainer.evaluate(eval_dataset)
        
        logger.info("=== 评估结果 ===")
        for key, value in results.items():
            logger.info(f"{key}: {value}")
        logger.info("================")
        
        return results
    
    def predict(self, text: str, max_length: int = 512) -> str:
        """使用模型进行预测"""
        import torch
        
        # 编码输入文本
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True
        )
        
        # 移动到GPU
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # 解码输出
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return generated_text


def create_trainer(model, tokenizer, config: Dict[str, Any]) -> ToolUseTrainer:
    """创建训练器"""
    return ToolUseTrainer(model, tokenizer, config)
