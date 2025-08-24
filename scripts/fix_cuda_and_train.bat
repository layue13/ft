@echo off
echo 修复CUDA环境并开始训练...

REM 检查Python环境
echo 检查Python环境...
python scripts/check_python.py
if errorlevel 1 (
    echo Python环境检查失败
    pause
    exit /b 1
)

REM 检查CUDA环境
echo 检查CUDA环境...
python scripts/check_cuda.py
if errorlevel 1 (
    echo CUDA环境检查失败，正在修复...
    python scripts/check_cuda.py
)

REM 检查Hugging Face登录
echo 检查Hugging Face登录状态...
huggingface-cli whoami
if errorlevel 1 (
    echo 请先登录Hugging Face:
    echo huggingface-cli login
    echo 然后申请Gemma3-1b-it模型访问权限
    pause
    exit /b 1
)

REM 准备数据
echo 准备训练数据...
uv run python scripts/prepare_data.py

REM 开始训练 (使用GTX 1080优化配置)
echo 开始训练 (GTX 1080优化配置)...
uv run python scripts/train.py --config configs/training_config_gtx1080.yaml

echo 训练完成！
pause
