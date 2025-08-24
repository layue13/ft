@echo off
echo 正在设置Windows CUDA环境...

REM 检查Python版本
python --version
if errorlevel 1 (
    echo 错误: 未找到Python，请先安装Python 3.9+
    echo 推荐安装Python 3.11或3.12版本
    echo 下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM 检查Python版本兼容性
python -c "import sys; version=sys.version_info; exit(0 if version.major==3 and 9<=version.minor<14 else 1)"
if errorlevel 1 (
    echo 错误: Python版本不兼容，需要Python 3.9-3.13，推荐使用Python 3.11或3.12
    pause
    exit /b 1
)

REM 检查CUDA
nvidia-smi
if errorlevel 1 (
    echo 警告: 未检测到NVIDIA GPU或CUDA驱动
    echo 请确保已安装NVIDIA驱动和CUDA Toolkit
    pause
)

REM 安装UV (如果未安装)
where uv >nul 2>nul
if errorlevel 1 (
    echo 正在安装UV包管理器...
    powershell -Command "irm https://astral.sh/uv/install.ps1 | iex"
    echo UV安装完成
    
    REM 刷新环境变量
    echo 请关闭当前命令行窗口，重新打开后继续运行此脚本
    pause
    exit /b 0
) else (
    echo UV已安装
)

REM 安装项目依赖
echo 正在安装项目依赖...
uv sync
if errorlevel 1 (
    echo 错误: 依赖安装失败
    echo 请检查网络连接和Python版本
    pause
    exit /b 1
)

REM 安装bitsandbytes (Windows版本)
echo 正在安装bitsandbytes...
uv add bitsandbytes
if errorlevel 1 (
    echo 警告: bitsandbytes安装失败，将使用CPU模式
    echo 这可能会影响训练性能
)

echo 环境设置完成！
echo.
echo 下一步:
echo 1. 确保已登录Hugging Face: huggingface-cli login
echo 2. 运行数据准备: uv run python scripts/prepare_data.py
echo 3. 开始训练: uv run python scripts/train.py --config configs/training_config_windows.yaml
pause
