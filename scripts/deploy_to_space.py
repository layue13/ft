#!/usr/bin/env python3
"""
部署到Hugging Face Space的脚本
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path


def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger(__name__)


def check_git_status():
    """检查Git状态"""
    logger = logging.getLogger(__name__)
    
    try:
        # 检查是否在Git仓库中
        result = subprocess.run(["git", "status"], capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("当前目录不是Git仓库")
            return False
        
        # 检查是否有未提交的更改
        result = subprocess.run(["git", "diff", "--quiet"], capture_output=True)
        if result.returncode != 0:
            logger.warning("有未提交的更改，建议先提交")
            return True
        
        logger.info("Git状态正常")
        return True
        
    except FileNotFoundError:
        logger.error("Git未安装")
        return False


def init_git_repo():
    """初始化Git仓库"""
    logger = logging.getLogger(__name__)
    
    try:
        # 初始化Git仓库
        subprocess.run(["git", "init"], check=True)
        logger.info("Git仓库初始化完成")
        
        # 添加所有文件
        subprocess.run(["git", "add", "."], check=True)
        logger.info("文件已添加到暂存区")
        
        # 提交
        subprocess.run(["git", "commit", "-m", "Initial commit: Gemma3-1b tool calling finetuning"], check=True)
        logger.info("初始提交完成")
        
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Git操作失败: {e}")
        return False


def setup_hf_repo(repo_name, username):
    """设置Hugging Face仓库"""
    logger = logging.getLogger(__name__)
    
    try:
        # 添加远程仓库
        remote_url = f"https://huggingface.co/{username}/{repo_name}"
        subprocess.run(["git", "remote", "add", "origin", remote_url], check=True)
        logger.info(f"远程仓库已添加: {remote_url}")
        
        # 推送代码
        subprocess.run(["git", "push", "-u", "origin", "main"], check=True)
        logger.info("代码已推送到Hugging Face Hub")
        
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"推送失败: {e}")
        return False


def create_space_config():
    """创建Space配置文件"""
    logger = logging.getLogger(__name__)
    
    # 检查必要文件是否存在
    required_files = [
        "app.py",
        "pyproject.toml",
        "configs/space_config.yaml",
        "src/",
        "scripts/"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        logger.error(f"缺少必要文件: {missing_files}")
        return False
    
    logger.info("所有必要文件都存在")
    return True


def print_next_steps(repo_name, username):
    """打印下一步操作指南"""
    logger = logging.getLogger(__name__)
    
    logger.info("\n" + "="*60)
    logger.info("🎉 部署准备完成!")
    logger.info("="*60)
    
    logger.info("\n📋 下一步操作:")
    logger.info("1. 访问 https://huggingface.co/spaces")
    logger.info("2. 点击 'Create new Space'")
    logger.info("3. 选择以下配置:")
    logger.info("   - Owner: 选择您的用户名")
    logger.info("   - Space name: 设置Space名称")
    logger.info("   - SDK: Gradio")
    logger.info("   - Python version: 3.9+")
    logger.info("   - Hardware: GPU (T4 Small 或更高)")
    
    logger.info("\n🔧 Space设置:")
    logger.info(f"- Repository: {username}/{repo_name}")
    logger.info("- 自动从pyproject.toml安装依赖")
    logger.info("- 主文件: app.py")
    
    logger.info("\n💰 成本控制:")
    logger.info("- T4 Small: $0.40/小时")
    logger.info("- 可以申请社区GPU资助")
    logger.info("- 建议先用小样本测试")
    
    logger.info("\n🚀 启动后:")
    logger.info("1. 等待Space构建完成")
    logger.info("2. 在Web界面中点击'开始训练'")
    logger.info("3. 监控训练进度")
    logger.info("4. 下载训练结果")
    
    logger.info("\n📞 获取帮助:")
    logger.info("- Space文档: https://huggingface.co/docs/hub/spaces")
    logger.info("- 社区论坛: https://discuss.huggingface.co")
    
    logger.info("="*60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="部署到Hugging Face Space")
    parser.add_argument(
        "--repo-name", 
        type=str, 
        default="gemma3-tool-finetuning",
        help="Hugging Face仓库名称"
    )
    parser.add_argument(
        "--username", 
        type=str, 
        required=True,
        help="Hugging Face用户名"
    )
    parser.add_argument(
        "--init-git", 
        action="store_true",
        help="初始化Git仓库"
    )
    parser.add_argument(
        "--push", 
        action="store_true",
        help="推送到Hugging Face Hub"
    )
    
    args = parser.parse_args()
    
    logger = setup_logging()
    logger.info("🚀 开始部署到Hugging Face Space")
    
    # 检查项目配置
    if not create_space_config():
        logger.error("项目配置检查失败")
        return
    
    # 检查Git状态
    if not check_git_status():
        if args.init_git:
            if not init_git_repo():
                logger.error("Git仓库初始化失败")
                return
        else:
            logger.error("请先初始化Git仓库或使用 --init-git 参数")
            return
    
    # 推送到HF Hub
    if args.push:
        if not setup_hf_repo(args.repo_name, args.username):
            logger.error("推送到Hugging Face Hub失败")
            return
    
    # 打印下一步指南
    print_next_steps(args.repo_name, args.username)


if __name__ == "__main__":
    main()
