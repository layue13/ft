@echo off
echo 中国网络环境训练脚本...

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

REM 测试镜像站连接
echo 测试镜像站连接...
python scripts/test_mirrors.py

REM 准备数据
echo 准备训练数据...
uv run python scripts/prepare_data.py

REM 开始训练 (使用镜像站)
echo 开始训练 (使用镜像站)...
uv run python scripts/train_with_mirror.py --config configs/training_config_china.yaml --mirror auto

echo 训练完成！
pause
