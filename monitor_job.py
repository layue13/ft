#!/usr/bin/env python3
"""
监控HF Jobs训练状态
"""

import os
import sys
from huggingface_hub import inspect_job, fetch_job_logs, login

def monitor_job(job_id):
    """监控训练任务状态"""
    
    # 检查HF Token
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("❌ 请设置 HF_TOKEN 环境变量")
        return
    
    login(token=hf_token)
    
    print(f"📋 监控任务: {job_id}")
    
    try:
        # 获取任务状态
        job_info = inspect_job(job_id=job_id)
        print(f"📊 状态: {job_info.status.stage}")
        print(f"⏱️ 创建时间: {job_info.created_at}")
        
        if job_info.status.message:
            print(f"💬 消息: {job_info.status.message}")
        
        # 获取日志
        print("\n📝 最新日志:")
        print("-" * 50)
        try:
            logs = fetch_job_logs(job_id=job_id)
            for log in logs:
                print(log, end='')
        except Exception as log_error:
            print(f"⚠️ 暂时无法获取日志: {log_error}")
        print("\n" + "-" * 50)
        
        # 根据状态给出提示
        if job_info.status.stage == "RUNNING":
            print("🔄 任务正在运行中...")
        elif job_info.status.stage == "COMPLETED":
            print("✅ 任务已完成！")
        elif job_info.status.stage == "ERROR":
            print("❌ 任务执行失败")
        
    except Exception as e:
        print(f"❌ 获取任务信息失败: {e}")

def main():
    if len(sys.argv) != 2:
        print("使用方法: python monitor_job.py <job_id>")
        sys.exit(1)
    
    job_id = sys.argv[1]
    monitor_job(job_id)

if __name__ == "__main__":
    main()