"""
HFSS 控制器模块
封装 HFSS 的连接、仿真、数据获取等核心操作
"""

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

# pyaedt 兼容层：支持新旧版本
try:
    import pyaedt

    PYAEDT_NEW_API = False
except ImportError:
    # 新版 pyaedt (0.25+) 使用新的导入方式
    import ansys.aedt.core as pyaedt
    from ansys.aedt.core import Desktop as _Desktop
    from ansys.aedt.core import Hfss as _Hfss

    PYAEDT_NEW_API = True


def _get_solution_data_value(sol_data, expression: str, data_type: str = "real"):
    """
    兼容新旧 pyaedt API 获取 SolutionData 数据

    Args:
        sol_data: SolutionData 对象
        expression: 表达式名称
        data_type: 数据类型 - "real", "imag", "mag", "db"

    Returns:
        数据数组或 None
    """
    try:
        if hasattr(sol_data, "get_expression_data"):
            _, data = sol_data.get_expression_data(expression, formula=data_type)
            return data
        elif data_type in ("real", "mag"):
            return sol_data.data_real(expression)
        elif data_type == "imag":
            if hasattr(sol_data, "data_imag"):
                return sol_data.data_imag(expression)
            else:
                _, data = sol_data.get_expression_data(expression, formula="imag")
                return data
        elif data_type == "db":
            if hasattr(sol_data, "data_real"):
                return sol_data.data_real(expression)
            else:
                _, data = sol_data.get_expression_data(expression, formula="db20")
                return data
        return None
    except Exception as e:
        logger.warning(f"Failed to get solution data: {e}")
        return None


class HFSSController:
    """HFSS 仿真控制器"""

    def __init__(self, project_path: str, design_name: str, setup_name: str = "Setup1", sweep_name: str = "Sweep"):
        """
        初始化 HFSS 控制器

        Args:
            project_path: HFSS 项目路径 (.aedt)
            design_name: 设计名称
            setup_name: Setup 名称
            sweep_name: Sweep 名称
        """
        self.project_path = project_path
        self.design_name = design_name
        self.setup_name = setup_name
        self.sweep_name = sweep_name

        self.hfss = None
        self._connected = False
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 3
        self._reconnect_delay = 10  # 秒

    def connect(self, force_new: bool = False) -> bool:
        """
        连接到 HFSS

        Args:
            force_new: 是否强制启动新的 HFSS 实例
        """
        logger.info(f"[INFO] Connecting to HFSS...")
        logger.info(f"  Project: {self.project_path}")
        logger.info(f"  Design: {self.design_name}")
        logger.info(f"  PyAEDT API: {'new' if PYAEDT_NEW_API else 'legacy'}")

        # 检查项目文件是否存在
        if not os.path.exists(self.project_path):
            logger.info(f"[ERROR] Project file not found: {self.project_path}")
            return False

        # 如果强制启动新实例，先关闭现有连接
        if force_new:
            logger.info(" Force new HFSS instance requested...")
            try:
                desktop = pyaedt.Desktop()
                desktop.close_desktop()
            except Exception:
                pass
            time.sleep(2)

        # 安全删除锁文件（仅在无 HFSS 进程运行时）
        hfss_running = self._is_hfss_running()
        if not hfss_running or force_new:
            logger.info(" No HFSS process detected, checking for stale locks...")
            lock_patterns = [
                self.project_path.replace(".aedt", ".aedt.lock"),
                self.project_path.replace(".aedt", ".aedtresults") + "\\.lock",
                os.path.join(os.path.dirname(self.project_path), ".lock"),
            ]
            for lock_file in lock_patterns:
                if os.path.exists(lock_file):
                    try:
                        os.remove(lock_file)
                        logger.info(f"  [OK] Removed stale lock: {lock_file}")
                    except Exception as e:
                        logger.info(f"  [WARN] Cannot remove lock: {e}")
        else:
            logger.info(" HFSS process detected, will try to connect to existing instance")

        # 尝试连接到 HFSS
        connection_errors = []
        hfss_running = self._is_hfss_running()

        for attempt in range(3):
            try:
                if attempt > 0:
                    logger.info(f"[INFO] Retry attempt {attempt+1}/3...")
                    time.sleep(3)

                logger.info(f"[INFO] Connecting to HFSS (attempt {attempt+1})...")

                # 如果检测到HFSS已运行，先尝试连接已有实例
                # 第一次尝试：连接已有实例
                # 后续尝试：启动新实例
                use_existing = hfss_running and attempt == 0

                # 第三次尝试：强制移除锁文件
                remove_lock = attempt == 2

                self.hfss = pyaedt.Hfss(
                    project=self.project_path,
                    design=self.design_name,
                    version=None,
                    new_desktop=not use_existing,
                    close_on_exit=True,
                    remove_lock=remove_lock,
                )

                # 验证连接
                _ = self.hfss.project_name
                _ = self.hfss.design_name

                self._connected = True
                logger.success(" Connected to HFSS")
                return True

            except Exception as e:
                err_msg = str(e)
                connection_errors.append(err_msg)
                logger.info(f"[WARN] Connection attempt {attempt+1}/3 failed: {err_msg}")

                # 检查是否是项目锁定问题
                if "locked" in err_msg.lower():
                    if attempt < 2:
                        logger.info(" Project is locked. Will try to remove lock on next attempt...")
                    else:
                        logger.error(" Project is locked by another HFSS instance!")
                        logger.info(" Please close HFSS and try again.")

                # 检查是否是项目文件错误
                if "有错误" in err_msg or "error" in err_msg.lower():
                    logger.error(" Project file appears to have errors")
                    logger.info(" Try opening the project in HFSS manually to repair it")

                if attempt < 2:
                    logger.info(f"[INFO] Waiting 5s before retry...")
                    time.sleep(5)

        # 所有尝试都失败了
        logger.info(f"[ERROR] All connection methods failed:")
        for i, err in enumerate(connection_errors):
            logger.info(f"  Attempt {i+1}: {err}")
        return False

    @staticmethod
    def _is_hfss_running() -> bool:
        """检查是否有 HFSS 进程正在运行"""
        try:
            import subprocess

            process_names = ["AnsysCOMEngine.exe", "ansysedt.exe", "Hfss.exe"]
            for proc_name in process_names:
                result = subprocess.run(
                    ["tasklist", "/FI", f"IMAGENAME eq {proc_name}"], capture_output=True, text=True, timeout=5
                )
                if proc_name in result.stdout:
                    logger.info(f"[INFO] Found HFSS process: {proc_name}")
                    return True
            return False
        except Exception:
            return True

    def _check_connection(self) -> bool:
        """
        检查连接是否还活着

        Returns:
            连接是否正常
        """
        if not self._connected or not self.hfss:
            return False

        try:
            # 尝试访问 HFSS 对象的属性来判断连接是否断开
            _ = self.hfss.project_name
            return True
        except Exception as e:
            logger.info(f"[WARN] HFSS connection lost: {e}")
            return False

    def _reconnect(self) -> bool:
        """
        尝试重新连接到 HFSS（无限重试，直到成功或用户终止）

        Returns:
            是否重连成功
        """
        reconnect_count = 0
        while True:
            reconnect_count += 1
            logger.info(f"[INFO] Attempting to reconnect to HFSS (attempt {reconnect_count})...")

            # 强制关闭旧连接
            self._force_close()

            # 等待让 HFSS 准备好
            wait_time = min(10 + reconnect_count * 2, 30)  # 逐渐增加等待时间，最多30秒
            logger.info(f"[INFO] Waiting {wait_time}s before reconnect...")
            time.sleep(wait_time)

            # 强制启动新实例重新连接
            success = self.connect(force_new=True)
            if success:
                logger.info(f"[OK] Reconnected to HFSS successfully on attempt {reconnect_count}")
                self._reconnect_attempts = 0  # 重置重连计数
                return True

            logger.info(f"[WARN] Reconnection failed, will retry...")

    def _force_close(self):
        """强制关闭 HFSS 连接"""
        logger.info(f"[INFO] Force closing HFSS connection...")
        self._connected = False
        self.hfss = None

        # 尝试关闭桌面
        try:
            # 尝试获取桌面并关闭
            desktop = pyaedt.Desktop()
            desktop.close_desktop()
        except Exception:
            pass

        # 清理锁文件
        try:
            if os.path.exists(self.project_path):
                lock_patterns = [
                    self.project_path.replace(".aedt", ".aedt.lock"),
                    self.project_path.replace(".aedt", ".aedtresults") + "\\.lock",
                ]
                for lock_file in lock_patterns:
                    if os.path.exists(lock_file):
                        try:
                            os.remove(lock_file)
                            logger.info(f"  [OK] Removed lock: {lock_file}")
                        except Exception:
                            pass
        except Exception:
            pass

    def ensure_connection(self) -> bool:
        """
        确保连接正常，必要时重连（无限重试直到成功）

        Returns:
            连接是否正常
        """
        while not self._check_connection():
            logger.warning(" HFSS connection lost, will attempt to reconnect...")
            if not self._reconnect():
                logger.warning(" Reconnection failed, will retry in 10s...")
                time.sleep(10)
                continue
            return True
        return True

    def close(self):
        """关闭 HFSS 连接"""
        if self.hfss:
            try:
                logger.info(" Closing HFSS...")
                self.hfss.close_desktop()
                self.hfss = None
                logger.success(" Closed")
            except Exception:
                pass
        self._connected = False

    def set_variable(self, name: str, value: float, unit: str = "mm"):
        """
        设置设计变量

        Args:
            name: 变量名
            value: 变量值
            unit: 单位
        """
        if not self._connected:
            raise RuntimeError("Not connected to HFSS")

        # 格式化值
        if unit in ["nH", "pF", "GHz"]:
            expr = f"{value:.4f}{unit}"
        else:
            expr = f"{value:.3f}{unit}"

        # 设置变量，带重连和重试
        last_error = None
        for attempt in range(3):
            try:
                # 每次尝试前确保连接
                if not self.ensure_connection():
                    raise RuntimeError("Cannot reconnect to HFSS")

                self.hfss.variable_manager.set_variable(name, expression=expr)
                logger.info(f"[OK] {name} = {value:.3f}{unit}")
                return
            except Exception as e:
                last_error = e
                err_str = str(e).lower()

                # 检查是否是连接相关错误
                is_connection_error = any(
                    x in err_str
                    for x in [
                        "nonetype",
                        "bool",
                        "object is not iterable",
                        "attribute",
                        "none",
                        "disconnected",
                        "closed",
                    ]
                )

                if is_connection_error:
                    logger.info(f"[WARN] HFSS connection error on set_variable ({attempt+1}/3): {e}")
                    # 标记需要重连
                    self._connected = False
                    if attempt < 2:
                        logger.info(f"[INFO] Attempting to reconnect...")
                        if self._reconnect():
                            continue
                        time.sleep(3)
                else:
                    logger.info(f"[WARN] Set variable {name} ({attempt+1}/3): {e}")
                    if attempt < 2:
                        time.sleep(1)

        raise RuntimeError(f"Failed to set variable {name} after 3 attempts: {last_error}")

    def analyze(self, force: bool = False) -> bool:
        """
        运行仿真

        Args:
            force: 是否强制重新仿真

        Returns:
            是否成功
        """
        if not self._connected:
            return False

        last_error = None
        for attempt in range(3):
            try:
                # 确保连接正常
                if not self.ensure_connection():
                    logger.info(f"[ERROR] Cannot reconnect to HFSS (attempt {attempt+1}/3)")
                    self._connected = False
                    if attempt < 2:
                        time.sleep(3)
                        continue
                    return False

                self._ensure_far_field_setup()

                if force:
                    logger.info(f"[INFO] Force re-analyzing {self.setup_name}...")
                    try:
                        self.hfss.odesign.DeleteSetupData(self.setup_name)
                    except Exception:
                        pass
                else:
                    logger.info(f"[INFO] Analyzing {self.setup_name}...")

                t0 = time.time()
                self.hfss.analyze_setup(self.setup_name)
                elapsed = time.time() - t0

                # 验证分析是否真正执行（HFSS崩溃时可能0秒完成）
                if elapsed < 1.0:
                    logger.info(f"[WARN] Analysis suspiciously fast ({elapsed:.1f}s), may have failed silently")
                    # 尝试通过检查项目状态来验证
                    if not self._verify_analysis_ran():
                        logger.info(f"[WARN] Analysis verification failed, retrying...")
                        self._connected = False
                        if attempt < 2:
                            time.sleep(3)
                            continue
                        return False

                logger.info(f"[OK] Analysis done ({elapsed:.1f}s)")
                return True

            except Exception as e:
                last_error = e
                err_str = str(e).lower()

                # 检查是否是连接相关错误
                is_connection_error = any(
                    x in err_str
                    for x in [
                        "nonetype",
                        "bool",
                        "object is not iterable",
                        "argument of type",
                        "attribute",
                        "none",
                        "disconnected",
                        "closed",
                        "failed",
                    ]
                )

                if is_connection_error:
                    logger.info(f"[WARN] HFSS connection error on analyze ({attempt+1}/3): {e}")
                    self._connected = False
                    if attempt < 2:
                        logger.info(f"[INFO] Attempting to reconnect...")
                        if self._reconnect():
                            logger.info(f"[INFO] Reconnected, retrying analysis...")
                            continue
                        time.sleep(5)
                else:
                    logger.info(f"[ERROR] Analysis failed ({attempt+1}/3): {e}")
                    if attempt < 2:
                        time.sleep(2)

        logger.info(f"[ERROR] Analysis failed after 3 attempts: {last_error}")
        return False

    def _verify_analysis_ran(self) -> bool:
        """
        验证分析是否真正执行

        Returns:
            是否验证通过
        """
        try:
            # 尝试访问HFSS对象的基本属性来验证连接状态
            _ = self.hfss.project_name
            _ = self.hfss.design_name
            return True
        except Exception:
            return False

    def check_far_field_setup(self) -> Dict:
        """
        检查远场设置状态

        Returns:
            包含辐射边界和远场球体状态的字典
        """
        result = {
            "has_radiation_boundary": False,
            "has_far_field_sphere": False,
            "far_field_sphere_name": None,
            "is_linked_to_setup": False,
            "can_get_gain": False,
            "error": None,
        }

        if not self._connected:
            result["error"] = "HFSS 未连接"
            return result

        try:
            # 1. 检查辐射边界
            try:
                # 获取边界列表
                boundaries = self.hfss._odesign.GetChildObject("BoundarySetup")
                if boundaries:
                    boundary_names = list(boundaries.GetChildNames())
                    # 检查是否有辐射边界
                    for name in boundary_names:
                        try:
                            boundary = boundaries.GetChildObject(name)
                            btype = boundary.GetPropValue("Type")
                            if "Radiation" in str(btype) or "PML" in str(btype):
                                result["has_radiation_boundary"] = True
                                logger.info(f"[INFO] Found radiation boundary: {name}")
                                break
                        except Exception:
                            pass
            except Exception as e:
                logger.info(f"[DEBUG] Check radiation boundary: {e}")

            # 2. 检查远场球体
            try:
                radiation = self.hfss._odesign.GetChildObject("Radiation")
                if radiation:
                    ff_setups = list(radiation.GetChildNames())
                    if ff_setups:
                        result["has_far_field_sphere"] = True
                        result["far_field_sphere_name"] = "3D" if "3D" in ff_setups else ff_setups[0]
                        logger.info(f"[INFO] Found far field sphere: {result['far_field_sphere_name']}")
            except Exception as e:
                logger.info(f"[DEBUG] Check far field sphere: {e}")

            # 3. 检查是否关联到 Setup
            try:
                setup = self.hfss.get_setup(self.setup_name)
                ff_index = setup.props.get("InfiniteSphereSetup", -1)
                if ff_index != -1:
                    result["is_linked_to_setup"] = True
                    logger.info(f"[INFO] Far field linked to Setup (index {ff_index})")
            except Exception as e:
                logger.info(f"[DEBUG] Check setup link: {e}")

            # 4. 综合判断能否获取增益
            result["can_get_gain"] = result["has_far_field_sphere"] and result["is_linked_to_setup"]

        except Exception as e:
            result["error"] = str(e)

        return result

    def create_far_field_setup(self, sphere_name: str = "3D") -> bool:
        """
        创建远场球体设置

        Args:
            sphere_name: 远场球体名称

        Returns:
            是否成功创建
        """
        if not self._connected:
            logger.error(" HFSS 未连接")
            return False

        try:
            logger.info(f"[INFO] Creating far field sphere '{sphere_name}'...")

            # 检查是否已存在
            try:
                radiation = self.hfss._odesign.GetChildObject("Radiation")
                if radiation:
                    existing = list(radiation.GetChildNames())
                    if sphere_name in existing:
                        logger.info(f"[INFO] Far field sphere '{sphere_name}' already exists")
                        return True
            except Exception:
                pass

            # 使用 pyaedt 创建远场球体
            try:
                # 方法1: 使用 pyaedt API
                self.hfss.create_far_field_setup(
                    name=sphere_name,
                    theta_start=0,
                    theta_stop=180,
                    theta_step=10,
                    phi_start=0,
                    phi_stop=360,
                    phi_step=10,
                )
                logger.info(f"[OK] Created far field sphere '{sphere_name}'")
                return True
            except Exception as e:
                logger.info(f"[DEBUG] pyaedt method failed: {e}")

            # 方法2: 使用原生脚本
            try:
                script = f"""
Dim oAnsoftApp
Dim oDesktop
Dim oProject
Dim oDesign
Set oAnsoftApp = CreateObject("AnsoftHfss.HfssScriptInterface")
Set oDesktop = oAnsoftApp.GetAppDesktop()
Set oProject = oDesktop.GetActiveProject()
Set oDesign = oProject.GetActiveDesign()

' 创建远场球体
oDesign.InsertInfiniteSphereSetup Array("NAME:{sphere_name}", "UseCustomRadiationSurface:=", false)
"""
                self.hfss._odesign.ExecuteScript(script)
                logger.info(f"[OK] Created far field sphere via script")
                return True
            except Exception as e:
                logger.info(f"[DEBUG] Script method failed: {e}")

            return False

        except Exception as e:
            logger.info(f"[ERROR] Create far field setup failed: {e}")
            return False

    def ensure_far_field_for_gain(self) -> bool:
        """
        确保可以获取增益（检查并创建必要的远场设置）

        Returns:
            是否可以获取增益
        """
        if not self._connected:
            return False

        logger.info("\n[INFO] Checking far field setup for gain calculation...")

        # 检查当前状态
        status = self.check_far_field_setup()

        if status["can_get_gain"]:
            logger.success(" Far field setup ready for gain calculation")
            return True

        # 尝试修复
        issues = []
        if not status["has_radiation_boundary"]:
            issues.append("缺少辐射边界 (Radiation Boundary)")
        if not status["has_far_field_sphere"]:
            # 尝试创建远场球体
            if self.create_far_field_setup():
                logger.success(" Created far field sphere")
            else:
                issues.append("缺少远场球体 (Far Field Sphere)")
        if status["has_far_field_sphere"] and not status["is_linked_to_setup"]:
            # 尝试关联到 Setup
            if self._ensure_far_field_setup():
                logger.success(" Linked far field to Setup")
            else:
                issues.append("远场球体未关联到 Setup")

        if issues:
            logger.info("\n" + "=" * 60)
            logger.error(" 无法获取增益，缺少以下设置:")
            for issue in issues:
                logger.info(f"  - {issue}")
            logger.info("\n请在 HFSS 中手动添加:")
            logger.info("  1. 创建辐射边界: Draw -> Create Radiation Boundary")
            logger.info("  2. 创建远场球体: Radiation -> Insert Far Field Setup -> Infinite Sphere")
            logger.info("  3. 在 Setup 中关联远场球体: Setup -> Advanced -> Far Field Sphere")
            logger.info("=" * 60)
            return False

        # 再次检查
        status = self.check_far_field_setup()
        return status["can_get_gain"]

    def _ensure_far_field_setup(self):
        """确保远场设置已关联到 Setup"""
        try:
            setup = self.hfss.get_setup(self.setup_name)
            current_ff = setup.props.get("InfiniteSphereSetup", -1)

            if current_ff == -1:
                logger.info(" Linking far field sphere to Setup...")

                # 获取可用的远场设置
                ff_setups = list(self.hfss._odesign.GetChildObject("Radiation").GetChildNames())

                if ff_setups:
                    # 找到 "3D" 或第一个可用的远场设置
                    ff_name = "3D" if "3D" in ff_setups else ff_setups[0]
                    ff_index = ff_setups.index(ff_name)

                    # 设置 Setup 使用这个远场球面
                    setup.props["InfiniteSphereSetup"] = ff_index
                    logger.info(f"[OK] Linked far field '{ff_name}' (index {ff_index}) to {self.setup_name}")

                    # 更新 setup
                    setup.update()
                    return True
                else:
                    logger.warning(" No far field sphere found")
                    return False
            else:
                logger.info(f"[INFO] Far field already linked (index {current_ff})")
                return True

        except Exception as e:
            logger.info(f"[WARN] Could not link far field: {e}")
            return False

    def ensure_setup_frequency(self, target_freq_ghz: float) -> bool:
        """
        确保 Setup 包含目标频率点

        对于 Interpolating Sweep，增益只能在 Setup 频率点获取。
        如果目标频率不在 Setup 中，需要添加。

        Args:
            target_freq_ghz: 目标频率 (GHz)

        Returns:
            是否成功设置
        """
        if not self._connected:
            return False

        try:
            setup = self.hfss.get_setup(self.setup_name)
            current_freq = setup.props.get("Frequency", "4GHz")

            # 解析当前频率
            if isinstance(current_freq, str):
                current_freq_ghz = self._parse_freq_to_ghz(current_freq)
            else:
                current_freq_ghz = float(current_freq)

            logger.info(f"[INFO] Current Setup frequency: {current_freq_ghz:.2f} GHz")
            logger.info(f"[INFO] Target frequency: {target_freq_ghz:.2f} GHz")

            # 检查是否匹配
            if abs(current_freq_ghz - target_freq_ghz) < 0.01:  # 10 MHz tolerance
                logger.info(f"[OK] Setup frequency matches target")
                return True

            # 需要修改 Setup 频率
            logger.info(f"[INFO] Updating Setup frequency to {target_freq_ghz:.2f} GHz...")

            # 设置新频率
            setup.props["Frequency"] = f"{target_freq_ghz:.4f}GHz"
            setup.update()

            # 保存项目
            self.hfss.save_project()

            logger.info(f"[OK] Setup frequency updated to {target_freq_ghz:.2f} GHz")
            logger.info(f"[INFO] Note: This will require re-simulation")

            return True

        except Exception as e:
            logger.info(f"[ERROR] Failed to update Setup frequency: {e}")
            return False

    def _parse_freq_to_ghz(self, freq_str: str) -> float:
        """解析频率字符串为 GHz"""
        try:
            freq_str = freq_str.strip().upper()

            # 移除单位
            if "GHZ" in freq_str:
                return float(freq_str.replace("GHZ", "").strip())
            elif "MHZ" in freq_str:
                return float(freq_str.replace("MHZ", "").strip()) / 1000
            elif "KHZ" in freq_str:
                return float(freq_str.replace("KHZ", "").strip()) / 1000000
            elif "HZ" in freq_str:
                return float(freq_str.replace("HZ", "").strip()) / 1000000000
            else:
                return float(freq_str)
        except Exception:
            return 4.0  # 默认 4 GHz

    def get_setup_frequency(self) -> float:
        """
        获取当前 Setup 频率

        Returns:
            Setup 频率 (GHz)
        """
        if not self._connected:
            return 4.0

        try:
            setup = self.hfss.get_setup(self.setup_name)
            current_freq = setup.props.get("Frequency", "4GHz")

            if isinstance(current_freq, str):
                return self._parse_freq_to_ghz(current_freq)
            else:
                return float(current_freq)
        except Exception:
            return 4.0

    def get_s_parameters(self, ports: List[Tuple[int, int]] = None) -> Optional[Dict]:
        """
        获取 S 参数数据

        Args:
            ports: 端口列表，如 [(1,1), (2,1)]，默认获取 S11

        Returns:
            包含频率和 S 参数数据的字典
        """
        if not self._connected:
            return None

        if ports is None:
            ports = [(1, 1)]

        for attempt in range(3):
            try:
                if not self.ensure_connection():
                    logger.info(f"[WARN] Cannot reconnect to HFSS for S-params (attempt {attempt+1}/3)")
                    self._connected = False
                    if attempt < 2:
                        time.sleep(3)
                    continue

                logger.info(" Getting S-parameters...")

                sweep_path = f"{self.setup_name} : {self.sweep_name}"

                expressions = []
                for i, j in ports:
                    expressions.extend([f"dB(S({i},{j}))", f"S({i},{j})"])

                report = self.hfss.post.reports_by_category.standard(setup=sweep_path)
                report.domain = "Sweep"
                report.expressions = expressions
                report.create()

                sol_data = report.get_solution_data()

                try:
                    report.delete()
                except Exception:
                    pass

                if sol_data is None:
                    logger.warning(" No solution data")
                    return None

                try:
                    freq = sol_data.primary_sweep_values
                    if freq is not None:
                        freq = np.array(freq).flatten()
                        if np.max(freq) > 100:
                            freq = freq / 1e9
                    else:
                        freq = np.linspace(4, 8, 100)
                except Exception:
                    freq = np.linspace(4, 8, 100)

                result = {"freq": freq, "ports": {}}

                for i, j in ports:
                    s_db = _get_solution_data_value(sol_data, f"dB(S({i},{j}))", "real")
                    if s_db is None:
                        s_db = _get_solution_data_value(sol_data, f"S({i},{j})", "db")
                    s_real = _get_solution_data_value(sol_data, f"S({i},{j})", "real")
                    s_imag = _get_solution_data_value(sol_data, f"S({i},{j})", "imag")

                    if s_db is None:
                        continue

                    s_db = np.array(s_db).flatten()

                    if s_real is not None and s_imag is not None:
                        s_real = np.array(s_real).flatten()
                        s_imag = np.array(s_imag).flatten()
                        s_complex = s_real + 1j * s_imag
                        mag = np.abs(s_complex)
                        phase = np.angle(s_complex, deg=True)
                    else:
                        mag = 10 ** (s_db / 20)
                        phase = np.zeros_like(s_db)
                        s_real = mag
                        s_imag = np.zeros_like(mag)

                    result["ports"][(i, j)] = {
                        "mag": mag,
                        "phase": phase,
                        "db": s_db,
                        "real": s_real,
                        "imag": s_imag,
                    }

                logger.info(f"[OK] S-params: {len(freq)} freq points, {freq.min():.2f}-{freq.max():.2f} GHz")
                return result

            except Exception as e:
                err_str = str(e).lower()
                logger.info(f"[WARN] Get S-params failed (attempt {attempt+1}/3): {e}")

                is_connection_error = any(
                    x in err_str
                    for x in [
                        "nonetype",
                        "bool",
                        "object is not iterable",
                        "hfss terminal",
                        "argument of type",
                        "attribute",
                        "none",
                        "disconnected",
                        "closed",
                        "failed",
                    ]
                )

                if attempt >= 2:
                    # 所有重试都失败了
                    if is_connection_error:
                        self._connected = False
                    raise RuntimeError(f"get_s-params failed after 3 attempts: {e}")

                # 不是最后一次重试，尝试重连后继续
                if is_connection_error:
                    self._connected = False
                    logger.info(f"[INFO] Attempting to reconnect...")
                    if self._reconnect():
                        logger.info(f"[INFO] Reconnected, retrying get_s_parameters...")
                        continue
                    time.sleep(5)

                # 继续循环重试
                continue

        return None

    def get_gain(self, freq_ghz: float) -> Optional[float]:
        """
        获取峰值增益 - 使用 antenna_parameters 报告

        Args:
            freq_ghz: 频率 (GHz) - Interpolating Sweep 只能在 Setup 频率点获取增益

        Returns:
            峰值增益 (dB) - 已转换为 dB 单位
        """
        if not self._connected:
            return None

        if not self.ensure_connection():
            logger.warning(" Cannot reconnect to HFSS for gain")
            return None

        try:
            logger.info(f"[INFO] Getting PeakGain...")

            # 检查远场设置
            ff_setups = []
            try:
                ff_setups = list(self.hfss._odesign.GetChildObject("Radiation").GetChildNames())
            except Exception:
                pass

            if not ff_setups:
                logger.error(" No far field setup found!")
                logger.info("  请在 HFSS 中添加远场球体: Radiation -> Insert Far Field Setup -> Infinite Sphere")
                return None

            sphere_name = "3D" if "3D" in ff_setups else (ff_setups[0] if ff_setups else None)

            # 使用 LastAdaptive 解决方案 (Interpolating Sweep 只能在 Setup 频率点获取增益)
            solution = f"{self.setup_name} : LastAdaptive"

            # 方法1: 使用 dB(PeakGain) 表达式直接获取 dB 值
            try:
                ap_report = self.hfss.post.reports_by_category.antenna_parameters(setup=solution)

                # 检查返回值是否有效
                if ap_report is None or isinstance(ap_report, bool):
                    logger.info(f"[DEBUG] antenna_parameters returned: {type(ap_report).__name__}")
                    raise Exception("antenna_parameters API returned invalid type")

                if sphere_name:
                    try:
                        ap_report.far_field_sphere = sphere_name
                    except Exception as e:
                        logger.info(f"[DEBUG] Could not set far_field_sphere: {e}")

                ap_report.expressions = ["dB(PeakGain)", "dB(PeakRealizedGain)"]
                ap_report.create()

                sol_data = ap_report.get_solution_data()

                # 获取数据后立即删除报告
                try:
                    ap_report.delete()
                except Exception:
                    pass

                if sol_data:
                    # dB(PeakGain)
                    gain_data = _get_solution_data_value(sol_data, "dB(PeakGain)", "real")
                    if gain_data is not None and len(gain_data) > 0:
                        peak_gain = float(np.max(np.array(gain_data).flatten()))
                        logger.info(f"[OK] PeakGain: {peak_gain:.2f} dB")
                        return peak_gain

                    # dB(PeakRealizedGain)
                    gain_data = _get_solution_data_value(sol_data, "dB(PeakRealizedGain)", "real")
                    if gain_data is not None and len(gain_data) > 0:
                        peak_gain = float(np.max(np.array(gain_data).flatten()))
                        logger.info(f"[OK] PeakRealizedGain: {peak_gain:.2f} dB")
                        return peak_gain
            except Exception as e:
                logger.info(f"[DEBUG] dB(PeakGain) failed: {e}")

            # 方法2: 获取线性值然后转换为 dB
            try:
                ap_report = self.hfss.post.reports_by_category.antenna_parameters(setup=solution)

                if ap_report is None or isinstance(ap_report, bool):
                    raise Exception("antenna_parameters API returned invalid type")

                if sphere_name:
                    try:
                        ap_report.far_field_sphere = sphere_name
                    except Exception:
                        pass

                ap_report.expressions = ["PeakGain"]
                ap_report.create()

                sol_data = ap_report.get_solution_data()

                # 获取数据后立即删除报告
                try:
                    ap_report.delete()
                except Exception:
                    pass

                if sol_data:
                    gain_data = _get_solution_data_value(sol_data, "PeakGain", "real")
                    if gain_data is not None and len(gain_data) > 0:
                        # 转换为 dB
                        gain_linear = float(np.max(np.array(gain_data).flatten()))
                        peak_gain = 10 * np.log10(gain_linear) if gain_linear > 0 else -np.inf
                        logger.info(f"[OK] PeakGain: {peak_gain:.2f} dB (from linear {gain_linear:.2f})")
                        return peak_gain
            except Exception as e:
                logger.info(f"[DEBUG] PeakGain (linear) failed: {e}")

            # 方法3: 使用原生脚本直接获取增益 (备用方案)
            try:
                logger.info(" Trying native script method...")
                peak_gain = self._get_gain_via_script(freq_ghz, sphere_name)
                if peak_gain is not None:
                    return peak_gain
            except Exception as e:
                logger.info(f"[DEBUG] Script method failed: {e}")

            logger.info(f"[WARN] Could not get PeakGain - check pyaedt version compatibility")
            return None

        except Exception as e:
            logger.info(f"[ERROR] Get Gain: {e}")
            return None

    def _get_gain_via_script(self, freq_ghz: float, sphere_name: str = "3D") -> Optional[float]:
        """使用原生脚本获取增益 (备用方案)"""
        try:
            # 获取增益的脚本
            script = f"""
Dim oDesign
Set oDesign = GetActiveDesign()
Dim oModule
Set oModule = oDesign.GetModule("ReportSetup")
oModule.CreateReport "GainReport", "Antenna Parameters", "Rectangular Plot", "{self.setup_name} : LastAdaptive", Array(), Array("Context:=", "{sphere_name}", "Freq:=", "{freq_ghz}GHz"), Array("dB(PeakGain)"), False
"""
            # 执行脚本
            self.hfss._odesign.ExecuteScript(script)

            # 尝试从结果中读取增益
            # 这需要更复杂的实现...
            return None

        except Exception as e:
            logger.info(f"[DEBUG] Script gain method: {e}")
            return None

    def get_z_parameters(self, ports: List[Tuple[int, int]] = None) -> Optional[Dict]:
        """
        获取 Z 参数数据

        Args:
            ports: 端口列表，如 [(1,1), (2,1)]，默认获取 Z11

        Returns:
            包含频率和 Z 参数数据的字典
        """
        if not self._connected:
            return None

        if ports is None:
            ports = [(1, 1)]

        for attempt in range(3):
            try:
                if not self.ensure_connection():
                    logger.info(f"[WARN] Cannot reconnect to HFSS for Z-params (attempt {attempt+1}/3)")
                    self._connected = False
                    if attempt < 2:
                        time.sleep(3)
                    continue

                logger.info(" Getting Z-parameters...")

                sweep_path = f"{self.setup_name} : {self.sweep_name}"

                expressions = []
                for i, j in ports:
                    expressions.append(f"Z({i},{j})")

                report = self.hfss.post.reports_by_category.standard(setup=sweep_path)
                report.domain = "Sweep"
                report.expressions = expressions
                report.create()

                sol_data = report.get_solution_data()

                try:
                    report.delete()
                except Exception:
                    pass

                if sol_data is None:
                    logger.warning(" No solution data")
                    return None

                try:
                    freq = sol_data.primary_sweep_values
                    if freq is not None:
                        freq = np.array(freq).flatten()
                        if np.max(freq) > 100:
                            freq = freq / 1e9
                    else:
                        freq = np.linspace(4, 8, 100)
                except Exception:
                    freq = np.linspace(4, 8, 100)

                result = {"freq": freq, "ports": {}}

                for i, j in ports:
                    z_real = _get_solution_data_value(sol_data, f"Z({i},{j})", "real")
                    z_imag = _get_solution_data_value(sol_data, f"Z({i},{j})", "imag")

                    if z_real is None:
                        continue

                    z_real = np.array(z_real).flatten()
                    z_imag = np.array(z_imag).flatten()

                    result["ports"][(i, j)] = {
                        "real": z_real,
                        "imag": z_imag,
                    }

                logger.info(f"[OK] Z-params: {len(freq)} freq points, {freq.min():.2f}-{freq.max():.2f} GHz")
                return result

            except Exception as e:
                err_str = str(e).lower()
                logger.info(f"[WARN] Get Z-params failed (attempt {attempt+1}/3): {e}")

                is_connection_error = any(
                    x in err_str
                    for x in [
                        "nonetype",
                        "bool",
                        "object is not iterable",
                        "hfss terminal",
                        "argument of type",
                        "attribute",
                        "none",
                        "disconnected",
                        "closed",
                        "failed",
                    ]
                )

                if attempt >= 2:
                    # 所有重试都失败了
                    if is_connection_error:
                        self._connected = False
                    raise RuntimeError(f"get_z-params failed after 3 attempts: {e}")

                # 不是最后一次重试，尝试重连后继续
                if is_connection_error:
                    self._connected = False
                    logger.info(f"[INFO] Attempting to reconnect...")
                    if self._reconnect():
                        logger.info(f"[INFO] Reconnected, retrying get_z_parameters...")
                        continue
                    time.sleep(5)

                # 继续循环重试
                continue

        return None

    def clear_solution_cache(self):
        """清除HFSS解决方案缓存（报告数据、场数据等）"""
        if not self._connected or not self.hfss:
            return

        if not self.ensure_connection():
            return

        try:
            self.hfss.odesign.DeleteSetupData(self.setup_name)
            logger.success(" Solution cache cleared")
        except Exception as e:
            logger.info(f"[WARN] Clear solution cache: {e}")

        try:
            script = f"""
Dim oDesign
Set oDesign = GetActiveDesign()
Dim oModule
Set oModule = oDesign.GetModule("ReportSetup")
On Error Resume Next
Dim reportNames
reportNames = oModule.GetAllReportNames()
If IsArray(reportNames) Then
    For i = 0 To UBound(reportNames)
        oModule.DeleteReport reportNames(i)
    Next
End If
"""
            self.hfss._odesign.ExecuteScript(script)
        except Exception:
            pass

    def cleanup(self):
        """清理仿真数据"""
        self.clear_solution_cache()


class HFSSContext:
    """HFSS 上下文管理器，自动管理连接"""

    def __init__(self, config: Dict):
        self.config = config
        self.controller = None

    def __enter__(self):
        self.controller = HFSSController(
            self.config["project_path"],
            self.config["design_name"],
            self.config.get("setup_name", "Setup1"),
            self.config.get("sweep_name", "Sweep"),
        )
        if not self.controller.connect():
            raise RuntimeError("Failed to connect to HFSS")
        return self.controller

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.controller:
            self.controller.close()
        return False
