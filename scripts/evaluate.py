#!/usr/bin/env python3
"""
评估脚本 - 优化版本
专门用于评估工具调用模型的性能
"""

import argparse
import logging
import sys
import os
import json
from typing import Dict, Any, List

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.utils import setup_logging, load_config, log_system_info
from src.model_config import load_model_and_tokenizer
from src.data_processor import create_data_processor, ToolCallValidator


class ToolCallEvaluator:
    """工具调用评估器"""
    
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
        self.validator = ToolCallValidator()
    
    def extract_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        """从文本中提取工具调用"""
        return self.validator.extract_tool_calls(text)
    
    def calculate_tool_call_metrics(self, predictions: List[str], targets: List[str]) -> Dict[str, float]:
        """计算工具调用相关指标"""
        total_samples = len(predictions)
        if total_samples == 0:
            return {
                "tool_call_accuracy": 0.0,
                "tool_name_accuracy": 0.0,
                "tool_args_accuracy": 0.0,
                "tool_call_f1": 0.0,
                "exact_match": 0.0
            }
        
        correct_tool_calls = 0
        correct_tool_names = 0
        correct_tool_args = 0
        exact_matches = 0
        
        for pred, target in zip(predictions, targets):
            pred_tools = self.extract_tool_calls(pred)
            target_tools = self.extract_tool_calls(target)
            
            # 检查工具调用数量
            if len(pred_tools) == len(target_tools):
                correct_tool_calls += 1
            
            # 检查工具名称和参数
            for pred_tool, target_tool in zip(pred_tools, target_tools):
                if pred_tool.get('name') == target_tool.get('name'):
                    correct_tool_names += 1
                
                if pred_tool.get('arguments') == target_tool.get('arguments'):
                    correct_tool_args += 1
            
            # 检查完全匹配
            if pred_tools == target_tools:
                exact_matches += 1
        
        # 计算F1分数
        precision = correct_tool_calls / total_samples if total_samples > 0 else 0
        recall = correct_tool_calls / total_samples if total_samples > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "tool_call_accuracy": correct_tool_calls / total_samples,
            "tool_name_accuracy": correct_tool_names / total_samples,
            "tool_args_accuracy": correct_tool_args / total_samples,
            "tool_call_f1": f1,
            "exact_match": exact_matches / total_samples
        }
    
    def evaluate_model(self, eval_dataset, max_samples: int = 100) -> Dict[str, Any]:
        """评估模型性能"""
        import torch
        
        logger.info(f"开始评估模型，最大样本数: {max_samples}")
        
        # 限制评估样本数
        if len(eval_dataset) > max_samples:
            eval_dataset = eval_dataset.select(range(max_samples))
        
        predictions = []
        targets = []
        
        self.model.eval()
        
        with torch.no_grad():
            for i, sample in enumerate(eval_dataset):
                if i % 10 == 0:
                    logger.info(f"处理样本 {i+1}/{len(eval_dataset)}")
                
                # 获取输入文本
                input_ids = sample["input_ids"]
                attention_mask = sample["attention_mask"]
                
                # 找到最后一个非padding token的位置
                last_token_idx = len(input_ids) - 1
                while last_token_idx >= 0 and input_ids[last_token_idx] == self.tokenizer.pad_token_id:
                    last_token_idx -= 1
                
                if last_token_idx < 0:
                    continue
                
                # 准备输入
                inputs = {
                    "input_ids": torch.tensor([input_ids[:last_token_idx+1]]),
                    "attention_mask": torch.tensor([attention_mask[:last_token_idx+1]])
                }
                
                # 移动到GPU
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                # 生成预测
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                
                # 解码预测和目标
                pred_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                target_text = self.tokenizer.decode(sample["labels"], skip_special_tokens=True)
                
                predictions.append(pred_text)
                targets.append(target_text)
        
        # 计算指标
        metrics = self.calculate_tool_call_metrics(predictions, targets)
        
        # 添加样本示例
        sample_results = []
        for i in range(min(5, len(predictions))):
            sample_results.append({
                "prediction": predictions[i],
                "target": targets[i],
                "pred_tools": self.extract_tool_calls(predictions[i]),
                "target_tools": self.extract_tool_calls(targets[i])
            })
        
        return {
            "metrics": metrics,
            "sample_results": sample_results,
            "total_samples": len(predictions)
        }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="评估Gemma3-1b工具调用模型")
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="模型路径"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/training_config.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--max_samples", 
        type=int, 
        default=100,
        help="最大评估样本数"
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        default=None,
        help="评估结果输出文件"
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
    
    # 记录系统信息
    log_system_info()
    
    logger.info("开始评估流程...")
    
    try:
        # 加载配置
        logger.info(f"加载配置文件: {args.config}")
        config = load_config(args.config)
        
        # 加载模型和分词器
        logger.info(f"加载模型: {args.model_path}")
        model, tokenizer = load_model_and_tokenizer(config)
        
        # 加载PEFT适配器
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.model_path)
        
        # 创建数据处理器
        logger.info("创建数据处理器...")
        data_processor = create_data_processor(tokenizer, config)
        
        # 准备评估数据
        logger.info("准备评估数据...")
        dataset_name = config["dataset"]["name"]
        datasets = data_processor.prepare_training_data(dataset_name)
        eval_dataset = datasets["eval"]
        
        logger.info(f"评估数据集大小: {len(eval_dataset)}")
        
        # 创建评估器
        logger.info("创建评估器...")
        evaluator = ToolCallEvaluator(tokenizer, model)
        
        # 执行评估
        logger.info("开始评估...")
        results = evaluator.evaluate_model(eval_dataset, args.max_samples)
        
        # 打印结果
        logger.info("=== 评估结果 ===")
        for key, value in results["metrics"].items():
            logger.info(f"{key}: {value:.4f}")
        logger.info(f"评估样本数: {results['total_samples']}")
        
        # 打印样本示例
        logger.info("=== 样本示例 ===")
        for i, sample in enumerate(results["sample_results"]):
            logger.info(f"样本 {i+1}:")
            logger.info(f"  预测: {sample['prediction'][:200]}...")
            logger.info(f"  目标: {sample['target'][:200]}...")
            logger.info(f"  预测工具: {sample['pred_tools']}")
            logger.info(f"  目标工具: {sample['target_tools']}")
            logger.info("")
        
        # 保存结果
        if args.output_file:
            logger.info(f"保存结果到: {args.output_file}")
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info("评估完成!")
        
    except Exception as e:
        logger.error(f"评估过程中发生错误: {e}")
        raise


if __name__ == "__main__":
    main()
