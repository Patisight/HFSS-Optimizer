#!/usr/bin/env python
"""
HFSS 天线优化程序 - 图形界面启动器
仅支持 PyQt6 版本
"""

import os
import sys
import traceback

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def check_dependencies():
    """检查依赖"""
    missing = []
    
    try:
        import PyQt6
    except ImportError:
        missing.append("PyQt6")
    
    try:
        import numpy
    except ImportError:
        missing.append("numpy")
    
    try:
        import pandas
    except ImportError:
        missing.append("pandas")
    
    return missing


def main():
    try:
        missing = check_dependencies()
        if missing:
            from loguru import logger
            logger.error(f"Missing dependencies: {', '.join(missing)}")
            logger.info(f"Run: pip install {' '.join(missing)}")
            sys.exit(1)
        
        from gui_pyqt6 import main
        main()
    except Exception as e:
        from loguru import logger
        logger.error(f"{e}")
        traceback.print_exc()
        input("\nPress Enter to exit...")
        sys.exit(1)


if __name__ == "__main__":
    main()
