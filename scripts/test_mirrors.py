#!/usr/bin/env python3
"""
镜像站测试脚本
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.mirror_utils import test_mirrors

if __name__ == "__main__":
    test_mirrors()
