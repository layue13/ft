#!/usr/bin/env python3
"""
使用HF Jobs CLI提交训练任务 - 基于官方文档的正确方法
"""

import os
import subprocess
from huggingface_hub import login

def submit_job_via_cli():
    """使用HF Jobs CLI提交任务"""
    
    # 检查HF Token
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("❌ 请设置 HF_TOKEN 环境变量")
        print("   export HF_TOKEN=your_token_here")
        return
    
    # 登录HF
    login(token=hf_token)
    print("✅ HF Hub 登录成功")
    
    print("🚀 使用HF Jobs CLI提交Gemma-3训练任务...")
    
    # 设置环境变量
    env = os.environ.copy()
    env["HF_TOKEN"] = hf_token
    env["HUGGING_FACE_HUB_TOKEN"] = hf_token
    env["PYTHONUNBUFFERED"] = "1"
    
    # 构建CLI命令
    cmd = [
        "hf", 
        "jobs", 
        "uv", 
        "run",
        "--flavor", "a10g-small",
        "--secrets", "HF_TOKEN",
        "--detach",
        "simple_train.py"
    ]
    
    print(f"📋 执行命令: {' '.join(cmd)}")
    
    try:
        # 执行命令
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            check=True
        )
        
        print("✅ 任务提交成功!")
        print("📤 输出:")
        print(result.stdout)
        
        if result.stderr:
            print("⚠️ 警告信息:")
            print(result.stderr)
            
    except subprocess.CalledProcessError as e:
        print(f"❌ 任务提交失败: {e}")
        print(f"📤 输出: {e.stdout}")
        print(f"📤 错误: {e.stderr}")
    except FileNotFoundError:
        print("❌ 找不到huggingface-cli命令")
        print("💡 请确保已安装: pip install huggingface_hub[cli]")

if __name__ == "__main__":
    submit_job_via_cli()