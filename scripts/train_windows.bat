@echo off
echo 开始Gemma3-1b工具调用微调训练...

REM 检查环境
echo 检查CUDA环境...
nvidia-smi
if errorlevel 1 (
    echo 错误: 未检测到NVIDIA GPU
    pause
    exit /b 1
)

REM 检查Hugging Face登录
echo 检查Hugging Face登录状态...
huggingface-cli whoami
if errorlevel 1 (
    echo 请先登录Hugging Face:
    echo huggingface-cli login
    pause
    exit /b 1
)

REM 开始训练
echo 开始训练...
uv run python scripts/train.py --config configs/training_config_windows.yaml

echo 训练完成！
pause
