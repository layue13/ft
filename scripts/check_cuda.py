#!/usr/bin/env python3
"""
CUDA环境检查脚本
"""

import subprocess
import sys
import os

def check_nvidia_driver():
    """检查NVIDIA驱动"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA驱动已安装")
            print(result.stdout)
            return True
        else:
            print("❌ NVIDIA驱动未安装或有问题")
            return False
    except FileNotFoundError:
        print("❌ 未找到nvidia-smi，请安装NVIDIA驱动")
        return False

def check_cuda_toolkit():
    """检查CUDA Toolkit"""
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ CUDA Toolkit已安装")
            print(result.stdout)
            return True
        else:
            print("❌ CUDA Toolkit未安装或有问题")
            return False
    except FileNotFoundError:
        print("❌ 未找到nvcc，请安装CUDA Toolkit")
        return False

def check_pytorch_cuda():
    """检查PyTorch CUDA支持"""
    try:
        import torch
        print(f"PyTorch版本: {torch.__version__}")
        
        if torch.cuda.is_available():
            print("✅ PyTorch支持CUDA")
            print(f"CUDA版本: {torch.version.cuda}")
            print(f"GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            return True
        else:
            print("❌ PyTorch不支持CUDA")
            print("可能的原因:")
            print("1. PyTorch安装的是CPU版本")
            print("2. CUDA版本不匹配")
            print("3. 需要重新安装PyTorch")
            return False
    except ImportError:
        print("❌ PyTorch未安装")
        return False

def install_pytorch_cuda():
    """安装支持CUDA的PyTorch"""
    print("正在安装支持CUDA的PyTorch...")
    
    # 检测CUDA版本
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            # 提取CUDA版本
            for line in result.stdout.split('\n'):
                if 'release' in line:
                    cuda_version = line.split('release')[1].split(',')[0].strip()
                    print(f"检测到CUDA版本: {cuda_version}")
                    
                    # 根据CUDA版本安装对应的PyTorch
                    if cuda_version.startswith('11'):
                        print("安装PyTorch for CUDA 11.8...")
                        os.system("uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
                    elif cuda_version.startswith('12'):
                        print("安装PyTorch for CUDA 12.1...")
                        os.system("uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
                    else:
                        print(f"不支持的CUDA版本: {cuda_version}")
                        print("请手动安装对应版本的PyTorch")
                    return True
    except:
        pass
    
    print("无法检测CUDA版本，安装默认CUDA版本...")
    os.system("uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    return True

def main():
    """主函数"""
    print("=== CUDA环境检查 ===")
    
    # 检查NVIDIA驱动
    driver_ok = check_nvidia_driver()
    
    # 检查CUDA Toolkit
    toolkit_ok = check_cuda_toolkit()
    
    # 检查PyTorch CUDA支持
    pytorch_ok = check_pytorch_cuda()
    
    print("\n=== 检查结果 ===")
    if driver_ok and toolkit_ok and pytorch_ok:
        print("✅ 所有检查通过，CUDA环境正常")
    else:
        print("❌ 发现问题，需要修复")
        
        if not pytorch_ok:
            print("\n正在修复PyTorch CUDA支持...")
            install_pytorch_cuda()
            print("\n请重新运行此脚本检查结果")

if __name__ == "__main__":
    main()
