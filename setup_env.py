#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HFSS 天线优化程序 - 一键环境配置工具
检测环境、安装依赖、验证配置、引导用户完成初始化
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Tuple


class Colors:
    """终端颜色"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    """打印标题"""
    print(f"\n{'='*60}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text}{Colors.END}")
    print('='*60)


def print_step(step: int, total: int, text: str):
    """打印步骤"""
    print(f"\n{Colors.BOLD}[{step}/{total}] {text}{Colors.END}")


def print_ok(text: str = "OK"):
    """打印成功"""
    print(f"{Colors.GREEN}[OK] {text}{Colors.END}")


def print_warn(text: str):
    """打印警告"""
    print(f"{Colors.YELLOW}[!] {text}{Colors.END}")


def print_error(text: str):
    """打印错误"""
    print(f"{Colors.RED}[X] {text}{Colors.END}")


def print_info(text: str):
    """打印信息"""
    print(f"{Colors.BLUE}    {text}{Colors.END}")


class EnvironmentSetup:
    """环境配置器"""
    
    # 必需的依赖包
    REQUIRED_PACKAGES = {
        'numpy': 'numpy>=1.21.0',
        'pandas': 'pandas>=1.3.0',
        'scipy': 'scipy>=1.7.0',
        'pyaedt': 'pyaedt>=0.6.70',
        'skopt': 'scikit-optimize>=0.9.0',
        'sklearn': 'scikit-learn>=1.0.0',
        'matplotlib': 'matplotlib>=3.5.0',
        'psutil': 'psutil>=5.8.0',
        'PyQt6': 'PyQt6>=6.0',
    }
    
    # 可选的依赖包
    OPTIONAL_PACKAGES = {
        'PIL': 'Pillow>=9.0.0',  # 图像处理
        'openpyxl': 'openpyxl>=3.0.0',  # Excel 支持
    }

    # 代理模型增强依赖（用于gpflow_svgp和incremental增量学习代理模型）
    SURROGATE_ENHANCED_PACKAGES = {
        'gpflow': 'gpflow>=2.0',  # 稀疏变分高斯过程（推荐用于复杂场景）
        'tensorflow': 'tensorflow>=2.10',  # GPflow依赖
    }
    
    def __init__(self):
        self.python_version = None
        self.pip_version = None
        self.hfss_paths = []
        self.missing_packages = []
        self.warnings = []
        self.errors = []
        
    def check_python(self) -> bool:
        """检测 Python 环境"""
        print_step(1, 6, "检测 Python 环境")
        
        version = sys.version_info
        self.python_version = f"{version.major}.{version.minor}.{version.micro}"
        
        print_info(f"Python 版本: {self.python_version}")
        print_info(f"Python 路径: {sys.executable}")
        print_info(f"系统平台: {platform.system()} {platform.release()}")
        
        # 检查版本范围
        if version.major == 3 and 8 <= version.minor <= 12:
            print_ok(f"Python 版本符合要求 (3.8-3.12)")
            return True
        elif version.major == 3 and version.minor > 12:
            print_warn(f"Python 版本过新，可能存在兼容性问题")
            return True
        else:
            print_error(f"Python 版本不符合要求，需要 3.8-3.12")
            self.errors.append("Python 版本不符合要求")
            return False
    
    def check_pip(self) -> bool:
        """检测 pip"""
        print_step(2, 6, "检测 pip")
        
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pip', '--version'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                self.pip_version = result.stdout.strip().split()[1]
                print_ok(f"pip 版本: {self.pip_version}")
                return True
            else:
                print_error("pip 不可用")
                self.errors.append("pip 不可用")
                return False
        except Exception as e:
            print_error(f"检测 pip 失败: {e}")
            self.errors.append(f"检测 pip 失败: {e}")
            return False
    
    def check_packages(self) -> Tuple[List[str], List[str]]:
        """检测已安装的包"""
        print_step(3, 6, "检测依赖包")
        
        installed = []
        missing = []
        
        for module, package in self.REQUIRED_PACKAGES.items():
            try:
                mod = __import__(module)
                version = getattr(mod, '__version__', 'unknown')
                
                # 特殊处理 pyaedt 版本
                if module == 'pyaedt':
                    print_ok(f"pyaedt ({version})")
                    self._check_pyaedt_version(version)
                else:
                    print_ok(f"{package.split('>=')[0]} ({version})")
                
                installed.append(module)
            except ImportError:
                print_error(f"{package} - 未安装")
                missing.append(package)
        
        self.missing_packages = missing
        return installed, missing
    
    def _check_pyaedt_version(self, version: str):
        """检查 pyaedt 版本兼容性"""
        try:
            # 解析版本号
            parts = version.split('.')
            major = int(parts[0]) if len(parts) > 0 else 0
            minor = int(parts[1]) if len(parts) > 1 else 0
            
            # HFSS 2023R1 (v231) 兼容的 pyaedt 版本
            # 0.6.x - 0.8.x 或 0.20+ 应该都可以
            
            if major == 0:
                if minor < 6:
                    print_warn(f"pyaedt {version} 版本过旧，建议升级到 0.6.70+")
                    self.warnings.append(f"pyaedt 版本过旧 ({version})")
                elif minor == 20:
                    print_info(f"pyaedt {version} - 注意: 0.20.x 版本 API 可能有变化")
                elif minor >= 6 and minor <= 8:
                    print_ok(f"pyaedt {version} 版本兼容")
                else:
                    print_info(f"pyaedt {version} - 版本较新，请测试兼容性")
            
        except Exception as e:
            print_info(f"无法解析 pyaedt 版本: {version}")
    
    def check_optional_packages(self) -> List[str]:
        """检测可选包"""
        print_info("\n可选依赖:")
        missing_optional = []
        
        for module, package in self.OPTIONAL_PACKAGES.items():
            try:
                __import__(module)
                print_ok(f"{package.split('>=')[0]} - 已安装")
            except ImportError:
                print_info(f"{package} - 未安装 (可选)")
                missing_optional.append(package)
        
        return missing_optional

    def check_surrogate_enhanced_packages(self) -> List[str]:
        """检测代理模型增强依赖（gpflow, tensorflow）

        这些包用于支持 gpflow_svgp 和 incremental 增量学习代理模型
        """
        print_info("\n代理模型增强依赖 (用于增量学习代理模型):")
        missing_surrogate = []

        for module, package in self.SURROGATE_ENHANCED_PACKAGES.items():
            try:
                __import__(module)
                print_ok(f"{package.split('>=')[0]} - 已安装")
            except ImportError:
                print_warn(f"{package} - 未安装")
                print_info(f"  → 如需使用 gpflow_svgp 或 incremental 代理模型，请安装此包")
                missing_surrogate.append(package)

        if missing_surrogate:
            print_info("\n安装命令: pip install gpflow tensorflow")
            print_info("或使用简化版: pip install gpflow")

        return missing_surrogate

    def find_hfss(self) -> List[str]:
        """检测 HFSS 安装路径"""
        print_step(4, 6, "检测 HFSS 安装")
        
        hfss_paths = []
        
        # 常见安装路径
        common_paths = [
            r"C:\Program Files\AnsysEM",
            r"C:\Program Files\Ansys",
            r"C:\AnsysEM",
            r"D:\Program Files\AnsysEM",
            r"D:\AnsysEM",
            r"E:\Program Files\AnsysEM",
        ]
        
        # 检查环境变量
        ansys_root = os.environ.get('ANSYS_ROOT', '')
        if ansys_root:
            common_paths.insert(0, ansys_root)
        
        # 扫描路径
        for base_path in common_paths:
            if os.path.exists(base_path):
                # 查找版本目录
                try:
                    for item in os.listdir(base_path):
                        version_path = os.path.join(base_path, item)
                        ansysedt = os.path.join(version_path, "Win64", "ansysedt.exe")
                        if os.path.exists(ansysedt):
                            hfss_paths.append(version_path)
                            print_ok(f"找到 HFSS: {version_path}")
                except PermissionError:
                    pass
        
        if not hfss_paths:
            print_warn("未找到 HFSS 安装")
            print_info("请确保 HFSS 已安装，或手动配置路径")
            self.warnings.append("未找到 HFSS 安装")
        else:
            print_info(f"共找到 {len(hfss_paths)} 个 HFSS 版本")
        
        self.hfss_paths = hfss_paths
        return hfss_paths
    
    def install_packages(self, packages: List[str]) -> bool:
        """安装缺失的包"""
        if not packages:
            print_ok("所有依赖已安装，无需额外安装")
            return True
        
        print_step(5, 6, "安装缺失的依赖包")
        print_info(f"将安装: {', '.join(packages)}")
        print_info("这可能需要几分钟时间...\n")
        
        # 先升级 pip
        print_info("升级 pip...")
        subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'],
            capture_output=True
        )
        
        # 安装依赖
        for package in packages:
            print_info(f"安装 {package}...")
            try:
                result = subprocess.run(
                    [sys.executable, '-m', 'pip', 'install', package],
                    capture_output=True, text=True, timeout=300
                )
                if result.returncode == 0:
                    print_ok(f"{package} 安装成功")
                else:
                    print_error(f"{package} 安装失败")
                    print_info(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
                    self.errors.append(f"{package} 安装失败")
            except subprocess.TimeoutExpired:
                print_error(f"{package} 安装超时")
                self.errors.append(f"{package} 安装超时")
            except Exception as e:
                print_error(f"{package} 安装异常: {e}")
                self.errors.append(f"{package} 安装异常: {e}")
        
        return len(self.errors) == 0
    
    def create_default_config(self) -> bool:
        """创建默认配置文件"""
        print_step(6, 6, "创建默认配置")
        
        config_path = Path(__file__).parent / "user_config.json"
        
        if config_path.exists():
            print_ok("配置文件已存在，跳过创建")
            return True
        
        # 使用正确的配置格式
        default_config = {
            "hfss": {
                "project_path": "",
                "design_name": "HFSSDesign1",
                "setup_name": "Setup1",
                "sweep_name": "Sweep"
            },
            "variables": [
                {
                    "name": "Rl",
                    "bounds": [10.0, 30.0],
                    "unit": "mm",
                    "precision": 2
                },
                {
                    "name": "Wm",
                    "bounds": [0.2, 1.5],
                    "unit": "mm",
                    "precision": 2
                }
            ],
            "objectives": [
                {
                    "type": "s_db",
                    "name": "S11_max",
                    "goal": -10.0,
                    "target": "minimize",
                    "weight": 1.0,
                    "freq_range": [5.1, 7.2],
                    "port": [1, 1],
                    "constraint": "max"
                }
            ],
            "algorithm": {
                "algorithm": "mopso",
                "population_size": 20,
                "n_generations": 30,
                "use_surrogate": True,
                "surrogate_type": "rf",
                "surrogate_min_samples": 6,
                "surrogate_threshold": 0.5,
                "inertia_weight": 0.4,
                "c1": 1.5,
                "c2": 1.5,
                "load_evaluations": None
            },
            "visualization": {
                "plot_interval": 5
            },
            "run": {
                "output_dir": "./optim_results",
                "clear_old_results": False
            }
        }
        
        import json
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=4, ensure_ascii=False)
            print_ok(f"创建默认配置: {config_path}")
            print_info("请编辑 user_config.json 配置你的项目和变量")
            return True
        except Exception as e:
            print_error(f"创建配置失败: {e}")
            return False
    
    def verify_installation(self) -> bool:
        """验证安装"""
        print_header("验证安装")
        
        success = True
        
        # 测试导入
        print("测试模块导入...")
        test_imports = [
            ('numpy', 'np.array([1,2,3])'),
            ('pandas', 'pd.DataFrame([1,2,3])'),
            ('scipy', 'from scipy import optimize'),
            ('matplotlib', 'import matplotlib.pyplot'),
            ('skopt', 'from skopt import Optimizer'),
            ('sklearn', 'from sklearn.gaussian_process import GaussianProcessRegressor'),
        ]
        
        for module, test_code in test_imports:
            try:
                exec(f"import {module}")
                print_ok(f"{module}")
            except ImportError as e:
                print_error(f"{module}: {e}")
                success = False
        
        # 测试 pyaedt
        print("\n测试 pyaedt...")
        try:
            import pyaedt
            print_ok(f"pyaedt ({pyaedt.__version__})")
        except ImportError as e:
            print_error(f"pyaedt: {e}")
            success = False
        
        return success
    
    def run_full_setup(self, install: bool = True, install_surrogate: bool = False):
        """运行完整配置流程"""
        print_header("HFSS 天线优化程序 - 一键环境配置")
        print(f"时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 1. 检测 Python
        self.check_python()
        
        # 2. 检测 pip
        self.check_pip()
        
        # 3. 检测依赖
        installed, missing = self.check_packages()
        
        # 检测可选包
        self.check_optional_packages()

        # 检测代理模型增强依赖
        surrogate_missing = self.check_surrogate_enhanced_packages()
        
        # 4. 检测 HFSS
        self.find_hfss()
        
        # 5. 安装缺失依赖
        if install and missing:
            self.install_packages(missing)
        
        # 5.5 安装代理模型增强依赖（可选）
        if install_surrogate and surrogate_missing:
            print_info("\n安装代理模型增强依赖...")
            self.install_packages(surrogate_missing)
        
        # 6. 创建默认配置
        self.create_default_config()
        
        # 验证安装
        if install:
            self.verify_installation()
        
        # 总结
        self.print_summary()
        
        return len(self.errors) == 0
    
    def print_summary(self):
        """打印总结"""
        print_header("配置总结")
        
        print(f"Python: {self.python_version}")
        print(f"pip: {self.pip_version or 'N/A'}")
        print(f"缺失包: {len(self.missing_packages)}")
        print(f"HFSS: {len(self.hfss_paths)} 个版本")
        
        if self.warnings:
            print(f"\n{Colors.YELLOW}警告:{Colors.END}")
            for w in self.warnings:
                print(f"  - {w}")
        
        if self.errors:
            print(f"\n{Colors.RED}错误:{Colors.END}")
            for e in self.errors:
                print(f"  - {e}")
            print(f"\n{Colors.RED}配置未完成，请解决上述问题后重试{Colors.END}")
        else:
            print(f"\n{Colors.GREEN}[OK] 环境配置完成！{Colors.END}")
            print("\n接下来可以:")
            print("  1. 编辑 user_config.json 配置项目路径和变量")
            print("  2. 运行 启动优化程序.bat 开始优化")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='HFSS 优化程序环境配置')
    parser.add_argument('--check', action='store_true', help='仅检测环境，不安装')
    parser.add_argument('--install', action='store_true', help='仅安装缺失依赖')
    parser.add_argument('--surrogate', action='store_true', help='安装代理模型增强依赖 (gpflow, tensorflow)')
    parser.add_argument('--full', action='store_true', help='完整安装（包括代理模型增强依赖）')
    
    args = parser.parse_args()
    
    setup = EnvironmentSetup()
    
    if args.check:
        # 仅检测
        setup.check_python()
        setup.check_pip()
        setup.check_packages()
        setup.find_hfss()
        setup.print_summary()
    elif args.install:
        # 仅安装
        _, missing = setup.check_packages()
        if missing:
            setup.install_packages(missing)
            setup.verify_installation()
        else:
            print_ok("所有依赖已安装")
    elif args.surrogate:
        # 仅安装代理模型依赖
        surrogate_missing = setup.check_surrogate_enhanced_packages()
        if surrogate_missing:
            setup.install_packages(surrogate_missing)
        else:
            print_ok("代理模型依赖已安装")
    elif args.full:
        # 完整安装（包括代理模型）
        setup.run_full_setup(install=True, install_surrogate=True)
    else:
        # 默认完整流程（不含代理模型）
        setup.run_full_setup(install=True)
    
    return 0 if len(setup.errors) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
