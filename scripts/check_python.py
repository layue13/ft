#!/usr/bin/env python3
"""
Python版本检查脚本
"""

import sys
import subprocess
import os

def check_python_version():
    """检查Python版本"""
    version = sys.version_info
    print(f"当前Python版本: {version.major}.{version.minor}.{version.micro}")
    
    if version.major != 3:
        print("❌ 错误: 需要Python 3.x版本")
        return False
    
    if version.minor < 9:
        print("❌ 错误: 需要Python 3.9或更高版本")
        return False
    
    if version.minor >= 14:
        print("❌ 错误: Python版本过高，需要Python 3.9-3.13")
        return False
    
    print("✅ Python版本检查通过")
    return True

def check_pip():
    """检查pip是否可用"""
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], 
                      check=True, capture_output=True)
        print("✅ pip可用")
        return True
    except subprocess.CalledProcessError:
        print("❌ pip不可用")
        return False

def check_uv():
    """检查UV是否可用"""
    try:
        subprocess.run(["uv", "--version"], check=True, capture_output=True)
        print("✅ UV已安装")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ UV未安装")
        return False

def install_uv():
    """安装UV"""
    print("正在安装UV...")
    try:
        if os.name == 'nt':  # Windows
            subprocess.run([
                "powershell", "-Command", 
                "irm https://astral.sh/uv/install.ps1 | iex"
            ], check=True)
        else:  # Unix/Linux
            subprocess.run([
                "curl", "-LsSf", "https://astral.sh/uv/install.sh", "|", "sh"
            ], check=True, shell=True)
        print("✅ UV安装成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ UV安装失败: {e}")
        return False

def main():
    """主函数"""
    print("=== Python环境检查 ===")
    
    # 检查Python版本
    if not check_python_version():
        sys.exit(1)
    
    # 检查pip
    if not check_pip():
        print("请先安装pip")
        sys.exit(1)
    
    # 检查UV
    if not check_uv():
        print("UV未安装，正在安装...")
        if not install_uv():
            print("UV安装失败，请手动安装")
            print("Windows: https://docs.astral.sh/uv/getting-started/installation/")
            sys.exit(1)
    
    print("\n=== 环境检查完成 ===")
    print("可以继续运行 setup_windows.bat")

if __name__ == "__main__":
    main()
