#!/usr/bin/env python3
"""
llama.cpp安装脚本
提供多种安装方式，避免编译问题
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def install_with_uv():
    """使用uv安装llama-cpp-python"""
    print("📦 使用uv安装llama-cpp-python...")
    
    try:
        subprocess.run([
            "uv", "add", "llama-cpp-python"
        ], check=True)
        print("✅ uv安装成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ uv安装失败: {e}")
        return False

def install_with_pip():
    """使用pip安装llama-cpp-python"""
    print("📦 使用pip安装llama-cpp-python...")
    
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "llama-cpp-python"
        ], check=True)
        print("✅ pip安装成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ pip安装失败: {e}")
        return False

def install_with_conda():
    """使用conda安装"""
    print("📦 使用conda安装...")
    
    try:
        subprocess.run([
            "conda", "install", "-c", "conda-forge", "llama-cpp-python", "-y"
        ], check=True)
        print("✅ conda安装成功")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ conda安装失败")
        return False

def install_with_brew():
    """使用Homebrew安装 (macOS)"""
    print("🍺 使用Homebrew安装...")
    
    try:
        subprocess.run([
            "brew", "install", "llama-cpp"
        ], check=True)
        print("✅ Homebrew安装成功")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Homebrew安装失败")
        return False

def download_prebuilt():
    """下载预编译版本"""
    print("📥 下载预编译版本...")
    
    system = platform.system().lower()
    machine = platform.machine().lower()
    
    print(f"🖥️ 系统: {system} {machine}")
    
    # 这里可以添加预编译版本的下载链接
    print("💡 请访问 https://github.com/ggml-org/llama.cpp/releases")
    print("   下载适合你系统的预编译版本")
    
    return False

def main():
    print("🚀 llama.cpp安装工具")
    print("=" * 40)
    
    print("\n🔧 选择安装方式:")
    print("1. uv安装 (推荐)")
    print("2. pip安装")
    print("3. conda安装")
    print("4. Homebrew安装 (macOS)")
    print("5. 下载预编译版本")
    print("6. 手动安装")
    
    choice = input("\n请选择 (默认1): ").strip() or "1"
    
    success = False
    
    if choice == "1":
        success = install_with_uv()
    elif choice == "2":
        success = install_with_pip()
    elif choice == "3":
        success = install_with_conda()
    elif choice == "4":
        success = install_with_brew()
    elif choice == "5":
        success = download_prebuilt()
    elif choice == "6":
        print("\n📝 手动安装说明:")
        print("1. 访问: https://github.com/ggml-org/llama.cpp")
        print("2. 按照README中的说明进行安装")
        print("3. 或者使用: uv add llama-cpp-python")
        return
    
    if success:
        print("\n✅ 安装成功！")
        print("\n🔧 验证安装:")
        try:
            import llama_cpp
            print("✅ llama-cpp-python可以正常导入")
        except ImportError:
            print("⚠️ 需要重启Python环境")
    else:
        print("\n❌ 安装失败")
        print("\n💡 其他选项:")
        print("1. 使用uv: uv add llama-cpp-python")
        print("2. 使用pip: pip install llama-cpp-python")
        print("3. 手动编译: https://github.com/ggml-org/llama.cpp")
        print("4. 使用预编译版本")

if __name__ == "__main__":
    main()
