"""
训练器模块 - 优化版本
"""

import logging
import os
import json
import re
from typing import Dict, Any, Optional, List
from transformers import Trainer, TrainingArguments
from peft import PeftModel
from .utils import get_device_info, create_output_dir
from .model_config import prepare_model_for_training


logger = logging.getLogger(__name__)


class ToolCallEvaluator:
    """工具调用评估器"""
    
    @staticmethod
    def extract_tool_calls(text: str) -> List[Dict[str, Any]]:
        """从文本中提取工具调用"""
        tool_calls = []
        pattern = r'<tool_call>(.*?)</tool_call>'
        matches = re.findall(pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                lines = match.strip().split('\n')
                tool_call = {}
                
                for line in lines:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        if key == 'name':
                            tool_call['name'] = value
                        elif key == 'arguments':
                            tool_call['arguments'] = json.loads(value)
                        elif key == 'result':
                            tool_call['result'] = value
                
                if 'name' in tool_call and 'arguments' in tool_call:
                    tool_calls.append(tool_call)
            except Exception as e:
                logger.warning(f"解析工具调用失败: {e}")
        
        return tool_calls
    
    @staticmethod
    def calculate_tool_call_accuracy(predictions: List[str], targets: List[str]) -> Dict[str, float]:
        """计算工具调用准确率"""
        total_samples = len(predictions)
        if total_samples == 0:
            return {"tool_call_accuracy": 0.0, "tool_name_accuracy": 0.0, "tool_args_accuracy": 0.0}
        
        correct_tool_calls = 0
        correct_tool_names = 0
        correct_tool_args = 0
        
        for pred, target in zip(predictions, targets):
            pred_tools = ToolCallEvaluator.extract_tool_calls(pred)
            target_tools = ToolCallEvaluator.extract_tool_calls(target)
            
            # 检查工具调用数量
            if len(pred_tools) == len(target_tools):
                correct_tool_calls += 1
            
            # 检查工具名称和参数
            for pred_tool, target_tool in zip(pred_tools, target_tools):
                if pred_tool.get('name') == target_tool.get('name'):
                    correct_tool_names += 1
                
                if pred_tool.get('arguments') == target_tool.get('arguments'):
                    correct_tool_args += 1
        
        return {
            "tool_call_accuracy": correct_tool_calls / total_samples,
            "tool_name_accuracy": correct_tool_names / total_samples,
            "tool_args_accuracy": correct_tool_args / total_samples
        }


class ToolUseTrainer:
    """工具调用训练器 - 优化版本"""
    
    def __init__(self, model, tokenizer, config: Dict[str, Any]):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.trainer = None
        self.evaluator = ToolCallEvaluator()
        
        # 准备模型进行训练
        self.model = prepare_model_for_training(self.model)
        
        # 创建输出目录
        create_output_dir(config["training"]["output_dir"])
        
        # 记录设备信息
        device_info = get_device_info()
        logger.info(f"设备信息: {device_info}")
    
    def create_trainer(self, train_dataset, eval_dataset) -> Trainer:
        """创建训练器 - 优化版本"""
        from .model_config import create_training_arguments
        
        training_args = create_training_arguments(self.config)
        
        # 优化日志配置
        training_args.logging_strategy = "steps"
        training_args.logging_first_step = True
        training_args.logging_steps = 10  # 减少日志频率
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=self._data_collator,
            compute_metrics=self._compute_metrics,
        )
        
        return self.trainer
    
    def _data_collator(self, features):
        """数据整理器 - 优化版本"""
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
    
    def _compute_metrics(self, eval_pred):
        """计算评估指标 - 新增工具调用指标"""
        import torch
        import numpy as np
        
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=-1)
        
        # 计算标准指标
        metrics = {}
        
        # 计算损失
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(torch.tensor(predictions), torch.tensor(labels))
        metrics["eval_loss"] = loss.item()
        
        # 计算准确率
        correct = (predictions == labels).sum()
        total = labels.size
        metrics["eval_accuracy"] = correct / total if total > 0 else 0
        
        # 计算工具调用指标（如果可能）
        try:
            # 解码一些样本进行工具调用分析
            sample_predictions = []
            sample_labels = []
            
            for i in range(min(10, len(predictions))):
                pred_text = self.tokenizer.decode(predictions[i], skip_special_tokens=True)
                label_text = self.tokenizer.decode(labels[i], skip_special_tokens=True)
                sample_predictions.append(pred_text)
                sample_labels.append(label_text)
            
            tool_metrics = self.evaluator.calculate_tool_call_accuracy(
                sample_predictions, sample_labels
            )
            metrics.update(tool_metrics)
            
        except Exception as e:
            logger.warning(f"计算工具调用指标失败: {e}")
        
        return metrics
    
    def train(self, train_dataset, eval_dataset) -> None:
        """开始训练 - 优化版本"""
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
        logger.info(f"评估步数: {trainer.args.eval_steps}")
        logger.info(f"保存步数: {trainer.args.save_steps}")
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
        """保存模型 - 优化版本"""
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
        
        # 保存模型信息
        model_info = {
            "model_type": "gemma3-tool-use",
            "base_model": self.config["model"]["name"],
            "training_config": self.config,
            "model_parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
        }
        
        with open(os.path.join(output_dir, "model_info.json"), "w", encoding="utf-8") as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)
        
        logger.info("模型保存完成!")
    
    def evaluate(self, eval_dataset) -> Dict[str, float]:
        """评估模型 - 优化版本"""
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
        """使用模型进行预测 - 优化版本"""
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
