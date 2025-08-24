#!/usr/bin/env python3
"""
支持镜像站的训练脚本
"""

import argparse
import logging
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

from src.mirror_utils import MirrorSelector
from scripts.train import main as train_main

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="支持镜像站的训练脚本")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/training_config_china.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--mirror", 
        type=str, 
        choices=["modelscope", "tsinghua", "huggingface", "auto"],
        default="auto",
        help="指定镜像站 (auto=自动检测)"
    )
    parser.add_argument(
        "--test-mirrors", 
        action="store_true",
        help="测试所有镜像站连接"
    )
    
    args = parser.parse_args()
    setup_logging()
    
    if args.test_mirrors:
        print("=== 测试镜像站连接 ===")
        selector = MirrorSelector()
        network_results = selector.detect_network_environment()
        
        print("\n=== 测试结果 ===")
        for mirror_name, is_accessible in network_results.items():
            status = "✅ 正常" if is_accessible else "❌ 失败"
            print(f"{mirror_name}: {status}")
        
        print("\n=== 网络环境检测 ===")
        is_china = selector.is_china_network()
        print(f"中国网络环境: {'是' if is_china else '否'}")
        
        print("\n=== 推荐镜像站 ===")
        best_mirror, config = selector.select_best_mirror()
        print(f"推荐使用: {config['name']} ({config['base_url']})")
        return
    
    # 检测并选择最佳镜像站
    selector = MirrorSelector()
    
    if args.mirror == "auto":
        print("自动检测最佳镜像站...")
        best_mirror, config = selector.select_best_mirror()
        print(f"选择镜像站: {config['name']} ({config['base_url']})")
    else:
        print(f"使用指定镜像站: {args.mirror}")
        best_mirror = args.mirror
    
    # 修改配置文件以使用选定的镜像站
    import yaml
    with open(args.config, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)
    
    # 添加镜像站配置
    if "mirror" not in config_data:
        config_data["mirror"] = {}
    config_data["mirror"]["name"] = best_mirror
    
    # 保存临时配置文件
    temp_config = f"temp_config_{best_mirror}.yaml"
    with open(temp_config, 'w', encoding='utf-8') as f:
        yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
    
    print(f"使用镜像站 {best_mirror} 开始训练...")
    
    try:
        # 调用原始训练脚本
        sys.argv = [sys.argv[0], "--config", temp_config]
        train_main()
    finally:
        # 清理临时配置文件
        if os.path.exists(temp_config):
            os.remove(temp_config)

if __name__ == "__main__":
    main()
