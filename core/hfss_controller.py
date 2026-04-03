"""
HFSS 控制器模块
封装 HFSS 的连接、仿真、数据获取等核心操作
"""
import os
import time
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple, Any


class HFSSController:
    """HFSS 仿真控制器"""
    
    def __init__(self, project_path: str, design_name: str, 
                 setup_name: str = "Setup1", sweep_name: str = "Sweep"):
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
    
    def connect(self) -> bool:
        """连接到 HFSS"""
        try:
            import pyaedt
            
            print(f"[INFO] Connecting to HFSS...")
            print(f"  Project: {self.project_path}")
            print(f"  Design: {self.design_name}")
            
            # 安全删除锁文件（仅在无 HFSS 进程运行时）
            if not self._is_hfss_running():
                lock_patterns = [
                    self.project_path.replace('.aedt', '.aedt.lock'),
                    self.project_path.replace('.aedt', '.aedtresults') + '\\.lock',
                    os.path.join(os.path.dirname(self.project_path), '.lock'),
                ]
                for lock_file in lock_patterns:
                    if os.path.exists(lock_file):
                        try:
                            os.remove(lock_file)
                            print(f"  [OK] Removed stale lock: {lock_file}")
                        except Exception as e:
                            print(f"  [WARN] Cannot remove lock: {e}")
            else:
                print("[INFO] HFSS process detected, skipping lock cleanup")
            
            # 尝试连接到已有的 HFSS 实例
            try:
                # 先尝试连接现有桌面
                # 不指定 solution_type，保持项目原有设置
                self.hfss = pyaedt.Hfss(
                    project=self.project_path,
                    design=self.design_name,
                    version=None,
                    new_desktop=False,  # 先尝试复用现有桌面
                    close_on_exit=False,
                )
                print("[OK] Connected to existing HFSS instance")
            except Exception:
                # 如果失败，启动新的 HFSS
                print("[INFO] Starting new HFSS instance...")
                self.hfss = pyaedt.Hfss(
                    project=self.project_path,
                    design=self.design_name,
                    version=None,
                    new_desktop=True,
                    close_on_exit=True,
                )
            
            if self.hfss:
                self._connected = True
                print("[OK] Connected to HFSS")
                return True
            return False
            
        except Exception as e:
            print(f"[ERROR] Connection failed: {e}")
            return False
    
    @staticmethod
    def _is_hfss_running() -> bool:
        """检查是否有 HFSS 进程正在运行"""
        try:
            import subprocess
            result = subprocess.run(
                ['tasklist', '/FI', 'IMAGENAME eq AnsysCOMEngine.exe'],
                capture_output=True, text=True, timeout=5
            )
            return 'AnsysCOMEngine.exe' in result.stdout
        except Exception:
            # 无法判断时保守返回 True，不删锁文件
            return True

    def close(self):
        """关闭 HFSS 连接"""
        if self.hfss:
            try:
                print("[INFO] Closing HFSS...")
                self.hfss.close_desktop()
                self.hfss = None
                print("[OK] Closed")
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
        
        try:
            # 格式化值
            if unit in ["nH", "pF", "GHz"]:
                expr = f"{value:.4f}{unit}"
            else:
                expr = f"{value:.3f}{unit}"
            
            self.hfss.variable_manager.set_variable(name, expression=expr)
            print(f"[OK] {name} = {value:.3f}{unit}")
            
        except Exception as e:
            print(f"[WARN] Set variable {name}: {e}")
    
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
        
        try:
            # 确保远场设置已关联到 Setup
            self._ensure_far_field_setup()
            
            if force:
                print(f"[INFO] Force re-analyzing {self.setup_name}...")
                try:
                    self.hfss.odesign.DeleteSetupData(self.setup_name)
                except Exception:
                    pass
            else:
                print(f"[INFO] Analyzing {self.setup_name}...")
            
            t0 = time.time()
            self.hfss.analyze_setup(self.setup_name)
            elapsed = time.time() - t0
            
            print(f"[OK] Analysis done ({elapsed:.1f}s)")
            return True
            
        except Exception as e:
            print(f"[ERROR] Analysis failed: {e}")
            return False
    
    def check_far_field_setup(self) -> Dict:
        """
        检查远场设置状态
        
        Returns:
            包含辐射边界和远场球体状态的字典
        """
        result = {
            'has_radiation_boundary': False,
            'has_far_field_sphere': False,
            'far_field_sphere_name': None,
            'is_linked_to_setup': False,
            'can_get_gain': False,
            'error': None
        }
        
        if not self._connected:
            result['error'] = "HFSS 未连接"
            return result
        
        try:
            # 1. 检查辐射边界
            try:
                # 获取边界列表
                boundaries = self.hfss._odesign.GetChildObject('BoundarySetup')
                if boundaries:
                    boundary_names = list(boundaries.GetChildNames())
                    # 检查是否有辐射边界
                    for name in boundary_names:
                        try:
                            boundary = boundaries.GetChildObject(name)
                            btype = boundary.GetPropValue('Type')
                            if 'Radiation' in str(btype) or 'PML' in str(btype):
                                result['has_radiation_boundary'] = True
                                print(f"[INFO] Found radiation boundary: {name}")
                                break
                        except Exception:
                            pass
            except Exception as e:
                print(f"[DEBUG] Check radiation boundary: {e}")
            
            # 2. 检查远场球体
            try:
                radiation = self.hfss._odesign.GetChildObject('Radiation')
                if radiation:
                    ff_setups = list(radiation.GetChildNames())
                    if ff_setups:
                        result['has_far_field_sphere'] = True
                        result['far_field_sphere_name'] = '3D' if '3D' in ff_setups else ff_setups[0]
                        print(f"[INFO] Found far field sphere: {result['far_field_sphere_name']}")
            except Exception as e:
                print(f"[DEBUG] Check far field sphere: {e}")
            
            # 3. 检查是否关联到 Setup
            try:
                setup = self.hfss.get_setup(self.setup_name)
                ff_index = setup.props.get('InfiniteSphereSetup', -1)
                if ff_index != -1:
                    result['is_linked_to_setup'] = True
                    print(f"[INFO] Far field linked to Setup (index {ff_index})")
            except Exception as e:
                print(f"[DEBUG] Check setup link: {e}")
            
            # 4. 综合判断能否获取增益
            result['can_get_gain'] = (
                result['has_far_field_sphere'] and 
                result['is_linked_to_setup']
            )
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def create_far_field_setup(self, sphere_name: str = '3D') -> bool:
        """
        创建远场球体设置
        
        Args:
            sphere_name: 远场球体名称
            
        Returns:
            是否成功创建
        """
        if not self._connected:
            print("[ERROR] HFSS 未连接")
            return False
        
        try:
            print(f"[INFO] Creating far field sphere '{sphere_name}'...")
            
            # 检查是否已存在
            try:
                radiation = self.hfss._odesign.GetChildObject('Radiation')
                if radiation:
                    existing = list(radiation.GetChildNames())
                    if sphere_name in existing:
                        print(f"[INFO] Far field sphere '{sphere_name}' already exists")
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
                    phi_step=10
                )
                print(f"[OK] Created far field sphere '{sphere_name}'")
                return True
            except Exception as e:
                print(f"[DEBUG] pyaedt method failed: {e}")
            
            # 方法2: 使用原生脚本
            try:
                script = f'''
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
'''
                self.hfss._odesign.ExecuteScript(script)
                print(f"[OK] Created far field sphere via script")
                return True
            except Exception as e:
                print(f"[DEBUG] Script method failed: {e}")
            
            return False
            
        except Exception as e:
            print(f"[ERROR] Create far field setup failed: {e}")
            return False
    
    def ensure_far_field_for_gain(self) -> bool:
        """
        确保可以获取增益（检查并创建必要的远场设置）
        
        Returns:
            是否可以获取增益
        """
        if not self._connected:
            return False
        
        print("\n[INFO] Checking far field setup for gain calculation...")
        
        # 检查当前状态
        status = self.check_far_field_setup()
        
        if status['can_get_gain']:
            print("[OK] Far field setup ready for gain calculation")
            return True
        
        # 尝试修复
        issues = []
        if not status['has_radiation_boundary']:
            issues.append("缺少辐射边界 (Radiation Boundary)")
        if not status['has_far_field_sphere']:
            # 尝试创建远场球体
            if self.create_far_field_setup():
                print("[OK] Created far field sphere")
            else:
                issues.append("缺少远场球体 (Far Field Sphere)")
        if status['has_far_field_sphere'] and not status['is_linked_to_setup']:
            # 尝试关联到 Setup
            if self._ensure_far_field_setup():
                print("[OK] Linked far field to Setup")
            else:
                issues.append("远场球体未关联到 Setup")
        
        if issues:
            print("\n" + "=" * 60)
            print("[ERROR] 无法获取增益，缺少以下设置:")
            for issue in issues:
                print(f"  - {issue}")
            print("\n请在 HFSS 中手动添加:")
            print("  1. 创建辐射边界: Draw -> Create Radiation Boundary")
            print("  2. 创建远场球体: Radiation -> Insert Far Field Setup -> Infinite Sphere")
            print("  3. 在 Setup 中关联远场球体: Setup -> Advanced -> Far Field Sphere")
            print("=" * 60)
            return False
        
        # 再次检查
        status = self.check_far_field_setup()
        return status['can_get_gain']
    
    def _ensure_far_field_setup(self):
        """确保远场设置已关联到 Setup"""
        try:
            setup = self.hfss.get_setup(self.setup_name)
            current_ff = setup.props.get('InfiniteSphereSetup', -1)
            
            if current_ff == -1:
                print("[INFO] Linking far field sphere to Setup...")
                
                # 获取可用的远场设置
                ff_setups = list(self.hfss._odesign.GetChildObject('Radiation').GetChildNames())
                
                if ff_setups:
                    # 找到 "3D" 或第一个可用的远场设置
                    ff_name = '3D' if '3D' in ff_setups else ff_setups[0]
                    ff_index = ff_setups.index(ff_name)
                    
                    # 设置 Setup 使用这个远场球面
                    setup.props['InfiniteSphereSetup'] = ff_index
                    print(f"[OK] Linked far field '{ff_name}' (index {ff_index}) to {self.setup_name}")
                    
                    # 更新 setup
                    setup.update()
                    return True
                else:
                    print("[WARN] No far field sphere found")
                    return False
            else:
                print(f"[INFO] Far field already linked (index {current_ff})")
                return True
                
        except Exception as e:
            print(f"[WARN] Could not link far field: {e}")
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
            current_freq = setup.props.get('Frequency', '4GHz')
            
            # 解析当前频率
            if isinstance(current_freq, str):
                current_freq_ghz = self._parse_freq_to_ghz(current_freq)
            else:
                current_freq_ghz = float(current_freq)
            
            print(f"[INFO] Current Setup frequency: {current_freq_ghz:.2f} GHz")
            print(f"[INFO] Target frequency: {target_freq_ghz:.2f} GHz")
            
            # 检查是否匹配
            if abs(current_freq_ghz - target_freq_ghz) < 0.01:  # 10 MHz tolerance
                print(f"[OK] Setup frequency matches target")
                return True
            
            # 需要修改 Setup 频率
            print(f"[INFO] Updating Setup frequency to {target_freq_ghz:.2f} GHz...")
            
            # 设置新频率
            setup.props['Frequency'] = f"{target_freq_ghz:.4f}GHz"
            setup.update()
            
            # 保存项目
            self.hfss.save_project()
            
            print(f"[OK] Setup frequency updated to {target_freq_ghz:.2f} GHz")
            print(f"[INFO] Note: This will require re-simulation")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to update Setup frequency: {e}")
            return False
    
    def _parse_freq_to_ghz(self, freq_str: str) -> float:
        """解析频率字符串为 GHz"""
        try:
            freq_str = freq_str.strip().upper()
            
            # 移除单位
            if 'GHZ' in freq_str:
                return float(freq_str.replace('GHZ', '').strip())
            elif 'MHZ' in freq_str:
                return float(freq_str.replace('MHZ', '').strip()) / 1000
            elif 'KHZ' in freq_str:
                return float(freq_str.replace('KHZ', '').strip()) / 1000000
            elif 'HZ' in freq_str:
                return float(freq_str.replace('HZ', '').strip()) / 1000000000
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
            current_freq = setup.props.get('Frequency', '4GHz')
            
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
        
        try:
            print("[INFO] Getting S-parameters...")
            
            sweep_path = f"{self.setup_name} : {self.sweep_name}"
            
            # 构建表达式
            expressions = []
            for (i, j) in ports:
                expressions.extend([f"dB(S({i},{j}))", f"S({i},{j})"])
            
            report = self.hfss.post.reports_by_category.standard(setup=sweep_path)
            report.domain = "Sweep"
            report.expressions = expressions
            report.create()
            
            sol_data = report.get_solution_data()
            
            # 获取数据后立即删除报告，避免累积
            try:
                report.delete()
            except Exception:
                pass

            if sol_data is None or not hasattr(sol_data, 'data_real'):
                print("[WARN] No solution data")
                return None
            
            # 获取频率
            try:
                freq = sol_data.primary_sweep_values
                if freq is not None:
                    freq = np.array(freq).flatten()
                    if np.max(freq) > 100:  # Hz -> GHz
                        freq = freq / 1e9
                else:
                    freq = np.linspace(4, 8, 100)
            except Exception:
                freq = np.linspace(4, 8, 100)

            result = {
                'freq': freq,
                'ports': {}
            }
            
            # 解析每个端口
            for (i, j) in ports:
                s_db = sol_data.data_real(f"dB(S({i},{j}))")
                s_real = sol_data.data_real(f"S({i},{j})")
                s_imag = sol_data.data_imag(f"S({i},{j})")
                
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
                    # 无法获取实部和虚部，从 dB 值估算幅值，相位设为 0
                    mag = 10 ** (s_db / 20)
                    phase = np.zeros_like(s_db)
                    # 估算的实部和虚部（相位为 0）
                    s_real = mag
                    s_imag = np.zeros_like(mag)
                
                result['ports'][(i, j)] = {
                    'mag': mag,
                    'phase': phase,
                    'db': s_db,
                    'real': s_real,
                    'imag': s_imag,
                }
            
            print(f"[OK] S-params: {len(freq)} freq points, {freq.min():.2f}-{freq.max():.2f} GHz")
            return result
            
        except Exception as e:
            print(f"[ERROR] Get S-params: {e}")
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
        
        try:
            print(f"[INFO] Getting PeakGain...")
            
            # 检查远场设置
            ff_setups = []
            try:
                ff_setups = list(self.hfss._odesign.GetChildObject('Radiation').GetChildNames())
            except Exception:
                pass
            
            if not ff_setups:
                print("[ERROR] No far field setup found!")
                print("  请在 HFSS 中添加远场球体: Radiation -> Insert Far Field Setup -> Infinite Sphere")
                return None
            
            sphere_name = '3D' if '3D' in ff_setups else (ff_setups[0] if ff_setups else None)
            
            # 使用 LastAdaptive 解决方案 (Interpolating Sweep 只能在 Setup 频率点获取增益)
            solution = f"{self.setup_name} : LastAdaptive"
            
            # 方法1: 使用 dB(PeakGain) 表达式直接获取 dB 值
            try:
                ap_report = self.hfss.post.reports_by_category.antenna_parameters(setup=solution)
                
                # 检查返回值是否有效
                if ap_report is None or isinstance(ap_report, bool):
                    print(f"[DEBUG] antenna_parameters returned: {type(ap_report).__name__}")
                    raise Exception("antenna_parameters API returned invalid type")
                
                if sphere_name:
                    try:
                        ap_report.far_field_sphere = sphere_name
                    except Exception as e:
                        print(f"[DEBUG] Could not set far_field_sphere: {e}")
                
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
                    gain_data = sol_data.data_real("dB(PeakGain)")
                    if gain_data is not None and len(gain_data) > 0:
                        peak_gain = float(np.max(np.array(gain_data).flatten()))
                        print(f"[OK] PeakGain: {peak_gain:.2f} dB")
                        return peak_gain
                    
                    # dB(PeakRealizedGain)
                    gain_data = sol_data.data_real("dB(PeakRealizedGain)")
                    if gain_data is not None and len(gain_data) > 0:
                        peak_gain = float(np.max(np.array(gain_data).flatten()))
                        print(f"[OK] PeakRealizedGain: {peak_gain:.2f} dB")
                        return peak_gain
            except Exception as e:
                print(f"[DEBUG] dB(PeakGain) failed: {e}")
            
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
                    gain_data = sol_data.data_real("PeakGain")
                    if gain_data is not None and len(gain_data) > 0:
                        # 转换为 dB
                        gain_linear = float(np.max(np.array(gain_data).flatten()))
                        peak_gain = 10 * np.log10(gain_linear) if gain_linear > 0 else -np.inf
                        print(f"[OK] PeakGain: {peak_gain:.2f} dB (from linear {gain_linear:.2f})")
                        return peak_gain
            except Exception as e:
                print(f"[DEBUG] PeakGain (linear) failed: {e}")
            
            # 方法3: 使用原生脚本直接获取增益 (备用方案)
            try:
                print("[INFO] Trying native script method...")
                peak_gain = self._get_gain_via_script(freq_ghz, sphere_name)
                if peak_gain is not None:
                    return peak_gain
            except Exception as e:
                print(f"[DEBUG] Script method failed: {e}")
            
            print(f"[WARN] Could not get PeakGain - check pyaedt version compatibility")
            return None
            
        except Exception as e:
            print(f"[ERROR] Get Gain: {e}")
            return None
    
    def _get_gain_via_script(self, freq_ghz: float, sphere_name: str = '3D') -> Optional[float]:
        """使用原生脚本获取增益 (备用方案)"""
        try:
            # 获取增益的脚本
            script = f'''
Dim oDesign
Set oDesign = GetActiveDesign()
Dim oModule
Set oModule = oDesign.GetModule("ReportSetup")
oModule.CreateReport "GainReport", "Antenna Parameters", "Rectangular Plot", "{self.setup_name} : LastAdaptive", Array(), Array("Context:=", "{sphere_name}", "Freq:=", "{freq_ghz}GHz"), Array("dB(PeakGain)"), False
'''
            # 执行脚本
            self.hfss._odesign.ExecuteScript(script)
            
            # 尝试从结果中读取增益
            # 这需要更复杂的实现...
            return None
            
        except Exception as e:
            print(f"[DEBUG] Script gain method: {e}")
            return None
    
    def get_z_parameters(self) -> Optional[Dict]:
        """获取 Z 参数"""
        if not self._connected:
            return None
        
        try:
            sweep_path = f"{self.setup_name} : {self.sweep_name}"
            
            report = self.hfss.post.reports_by_category.standard(setup=sweep_path)
            report.domain = "Sweep"
            report.expressions = ["Z(1,1)"]
            report.create()
            
            sol_data = report.get_solution_data()
            
            # 获取数据后立即删除报告
            try:
                report.delete()
            except Exception:
                pass
            
            if sol_data is None:
                return None
            
            z_real = sol_data.data_real("Z(1,1)")
            z_imag = sol_data.data_imag("Z(1,1)")
            
            if z_real is None:
                return None
            
            z_real = np.array(z_real).flatten()
            z_imag = np.array(z_imag).flatten()
            
            # 使用 Sweep 的实际频率，而非硬编码
            try:
                freq = np.array(sol_data.primary_sweep_values).flatten()
                if np.max(freq) > 100:  # Hz -> GHz
                    freq = freq / 1e9
            except Exception:
                freq = np.linspace(4, 8, len(z_real))
            
            return {
                'freq': freq,
                'z_real': z_real,
                'z_imag': z_imag,
            }
            
        except Exception as e:
            print(f"[WARN] Get Z-params: {e}")
            return None
    
    def clear_solution_cache(self):
        """清除HFSS解决方案缓存（报告数据、场数据等）

        调用此方法可以释放HFSS内存中缓存的仿真数据，
        避免长时间运行后HFSS变得卡顿。
        在每次仿真迭代结束后调用。
        """
        if not self._connected or not self.hfss:
            return

        try:
            self.hfss.odesign.DeleteSetupData(self.setup_name)
            print("[OK] Solution cache cleared")
        except Exception as e:
            print(f"[WARN] Clear solution cache: {e}")

        try:
            script = f'''
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
'''
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
            self.config['project_path'],
            self.config['design_name'],
            self.config.get('setup_name', 'Setup1'),
            self.config.get('sweep_name', 'Sweep'),
        )
        if not self.controller.connect():
            raise RuntimeError("Failed to connect to HFSS")
        return self.controller
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.controller:
            self.controller.close()
        return False