"""
镜像站工具模块
用于检测网络环境并选择合适的Hugging Face镜像站
"""

import yaml
import requests
import time
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class MirrorSelector:
    """镜像站选择器"""
    
    def __init__(self, config_path: str = "configs/mirror_config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.selected_mirror = None
        
    def _load_config(self) -> Dict:
        """加载镜像站配置"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"镜像站配置文件 {self.config_path} 不存在，使用默认配置")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            "mirrors": {
                "hf_mirror": {
                    "name": "HF Mirror (标准镜像站)",
                    "base_url": "https://hf-mirror.com",
                    "models_path": "",
                    "datasets_path": "",
                    "enabled": True,
                    "priority": 1
                },
                "modelscope": {
                    "name": "ModelScope (阿里云)",
                    "base_url": "https://modelscope.cn",
                    "models_path": "/models",
                    "datasets_path": "/datasets",
                    "enabled": True,
                    "priority": 2
                },
                "huggingface": {
                    "name": "Hugging Face官方",
                    "base_url": "https://huggingface.co",
                    "models_path": "",
                    "datasets_path": "",
                    "enabled": True,
                    "priority": 3
                }
            },
            "network_detection": {
                "enabled": True,
                "timeout": 5
            },
            "auto_select": {
                "enabled": True,
                "prefer_china_mirrors": True,
                "fallback_to_official": True
            }
        }
    
    def test_network_connectivity(self, url: str, timeout: int = 5) -> bool:
        """测试网络连接"""
        try:
            response = requests.get(url, timeout=timeout)
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"网络连接测试失败 {url}: {e}")
            return False
    
    def detect_network_environment(self) -> Dict[str, bool]:
        """检测网络环境"""
        results = {}
        timeout = self.config.get("network_detection", {}).get("timeout", 5)
        
        for mirror_name, mirror_config in self.config["mirrors"].items():
            if not mirror_config.get("enabled", True):
                continue
                
            base_url = mirror_config["base_url"]
            logger.info(f"测试镜像站连接: {mirror_config['name']} ({base_url})")
            
            is_accessible = self.test_network_connectivity(base_url, timeout)
            results[mirror_name] = is_accessible
            
            if is_accessible:
                logger.info(f"✅ {mirror_config['name']} 连接正常")
            else:
                logger.warning(f"❌ {mirror_config['name']} 连接失败")
        
        return results
    
    def select_best_mirror(self, force_mirror: Optional[str] = None) -> Tuple[str, Dict]:
        """选择最佳镜像站"""
        if force_mirror and force_mirror in self.config["mirrors"]:
            mirror_config = self.config["mirrors"][force_mirror]
            logger.info(f"使用指定镜像站: {mirror_config['name']}")
            return force_mirror, mirror_config
        
        # 检测网络环境
        network_results = self.detect_network_environment()
        
        # 按优先级排序镜像站
        sorted_mirrors = sorted(
            self.config["mirrors"].items(),
            key=lambda x: x[1].get("priority", 999)
        )
        
        # 选择第一个可用的镜像站
        for mirror_name, mirror_config in sorted_mirrors:
            if not mirror_config.get("enabled", True):
                continue
                
            if network_results.get(mirror_name, False):
                logger.info(f"选择镜像站: {mirror_config['name']}")
                self.selected_mirror = mirror_name
                return mirror_name, mirror_config
        
        # 如果没有可用的镜像站，返回官方源
        logger.warning("所有镜像站都不可用，使用官方源")
        official_config = self.config["mirrors"]["huggingface"]
        return "huggingface", official_config
    
    def get_model_url(self, model_name: str, mirror_name: Optional[str] = None) -> str:
        """获取模型URL"""
        if not mirror_name:
            mirror_name, mirror_config = self.select_best_mirror()
        else:
            mirror_config = self.config["mirrors"][mirror_name]
        
        base_url = mirror_config["base_url"]
        models_path = mirror_config.get("models_path", "")
        
        # 处理不同的镜像站URL格式
        if mirror_name == "hf_mirror":
            # HF Mirror格式: https://hf-mirror.com/google/gemma-3-1b-it
            return f"{base_url}/{model_name}"
        elif mirror_name == "modelscope":
            # ModelScope格式: https://modelscope.cn/models/google/gemma-3-1b-it
            return f"{base_url}{models_path}/{model_name}"
        elif mirror_name == "tsinghua":
            # 清华镜像格式: https://mirrors.tuna.tsinghua.edu.cn/hugging-face-models/google/gemma-3-1b-it
            return f"{base_url}{models_path}/{model_name}"
        else:
            # Hugging Face官方格式: https://huggingface.co/google/gemma-3-1b-it
            return f"{base_url}/{model_name}"
    
    def get_dataset_url(self, dataset_name: str, mirror_name: Optional[str] = None) -> str:
        """获取数据集URL"""
        if not mirror_name:
            mirror_name, mirror_config = self.select_best_mirror()
        else:
            mirror_config = self.config["mirrors"][mirror_name]
        
        base_url = mirror_config["base_url"]
        datasets_path = mirror_config.get("datasets_path", "")
        
        # 处理不同的镜像站URL格式
        if mirror_name == "hf_mirror":
            # HF Mirror格式: https://hf-mirror.com/datasets/shawhin/tool-use-finetuning
            return f"{base_url}/datasets/{dataset_name}"
        elif mirror_name == "modelscope":
            # ModelScope格式: https://modelscope.cn/datasets/shawhin/tool-use-finetuning
            return f"{base_url}{datasets_path}/{dataset_name}"
        elif mirror_name == "tsinghua":
            # 清华镜像格式: https://mirrors.tuna.tsinghua.edu.cn/hugging-face-models/datasets--shawhin--tool-use-finetuning
            return f"{base_url}{datasets_path}/datasets--{dataset_name.replace('/', '--')}"
        else:
            # Hugging Face官方格式: https://huggingface.co/datasets/shawhin/tool-use-finetuning
            return f"{base_url}/datasets/{dataset_name}"
    
    def is_china_network(self) -> bool:
        """检测是否在中国网络环境"""
        # 测试几个中国特有的网站
        china_test_urls = [
            "https://www.baidu.com",
            "https://www.qq.com",
            "https://www.taobao.com"
        ]
        
        for url in china_test_urls:
            if self.test_network_connectivity(url, timeout=3):
                return True
        
        return False

def get_mirror_selector() -> MirrorSelector:
    """获取镜像站选择器实例"""
    return MirrorSelector()

def test_mirrors():
    """测试所有镜像站"""
    selector = MirrorSelector()
    
    print("=== 镜像站连接测试 ===")
    network_results = selector.detect_network_environment()
    
    print("\n=== 测试结果 ===")
    for mirror_name, is_accessible in network_results.items():
        status = "✅ 正常" if is_accessible else "❌ 失败"
        print(f"{mirror_name}: {status}")
    
    print("\n=== 网络环境检测 ===")
    is_china = selector.is_china_network()
    print(f"中国网络环境: {'是' if is_china else '否'}")
    
    print("\n=== 推荐镜像站 ===")
    best_mirror, config = selector.select_best_mirror()
    print(f"推荐使用: {config['name']} ({config['base_url']})")

if __name__ == "__main__":
    test_mirrors()
