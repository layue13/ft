# Windows CUDA环境安装指南

## 快速开始

### 1. 安装Python
- 下载并安装 [Python 3.11](https://www.python.org/downloads/release/python-3116/) 或 [Python 3.12](https://www.python.org/downloads/release/python-3120/)
- 确保勾选"Add Python to PATH"
- 安装完成后重启命令行

### 2. 检查环境
```cmd
python scripts/check_python.py
```

### 3. 安装项目依赖
```cmd
scripts/setup_windows.bat
```

## 详细步骤

### 步骤1: 安装Python
1. 访问 [Python官网](https://www.python.org/downloads/)
2. 下载Python 3.11.6或3.12.0
3. 运行安装程序，**重要**: 勾选"Add Python to PATH"
4. 安装完成后重启命令行

### 步骤2: 验证Python安装
```cmd
python --version
```
应该显示类似：`Python 3.11.6` 或 `Python 3.12.0`

### 步骤3: 安装UV包管理器
```cmd
powershell -Command "irm https://astral.sh/uv/install.ps1 | iex"
```
安装完成后重启命令行

### 步骤4: 克隆项目
```cmd
git clone git@github.com:layue13/ft.git
cd ft
```

### 步骤5: 安装项目依赖
```cmd
uv sync
```

### 步骤6: 安装bitsandbytes (可选)
```cmd
uv add bitsandbytes
```

## 常见问题解决

### 问题1: "No interpreter found for Python 3.13.7"
**解决方案**:
1. 卸载Python 3.13.7
2. 安装Python 3.11.6或3.12.0
3. 确保勾选"Add Python to PATH"
4. 重启命令行

### 问题2: "uv command not found"
**解决方案**:
1. 重新安装UV: `powershell -Command "irm https://astral.sh/uv/install.ps1 | iex"`
2. 重启命令行
3. 验证安装: `uv --version`

### 问题3: 网络连接问题
**解决方案**:
1. 使用VPN或代理
2. 配置pip镜像源:
```cmd
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

### 问题4: CUDA相关错误
**解决方案**:
1. 安装NVIDIA驱动
2. 安装CUDA Toolkit 11.8或12.1
3. 重启系统

## 环境要求

- **操作系统**: Windows 10/11
- **Python**: 3.9-3.13 (推荐3.11或3.12)
- **GPU**: NVIDIA GPU with 8GB+ VRAM
- **内存**: 16GB+ RAM
- **存储**: 10GB+ 可用空间

## 验证安装

运行以下命令验证安装：
```cmd
python scripts/check_python.py
```

如果所有检查都通过，就可以开始训练了：
```cmd
scripts/train_windows.bat
```
