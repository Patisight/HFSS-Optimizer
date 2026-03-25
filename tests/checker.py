#!/usr/bin/env python
"""
HFSS 项目自检模块

自检内容：
1. 变量边界检查
2. 随机采样测试（检测模型是否会出错）
3. 仿真设置检查
4. 远场配置检查

在优化开始前运行，避免中途出错
"""
import sys
import os
import random
import time
import numpy as np
from typing import Dict, List, Tuple, Optional

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


class ProjectChecker:
    """
    项目自检器
    
    检查项目配置和变量设置是否合理，避免优化中途出错
    """
    
    def __init__(self, config: Dict, n_samples: int = 10):
        self.config = config
        self.n_samples = n_samples
        self.hfss = None
        self.original_values = {}  # 保存原始变量值
        self.results = {
            'passed': [],
            'warnings': [],
            'errors': [],
            'details': {},
            'failed_params': []  # 失败的参数组合
        }
    
    def run_all_checks(self, progress_callback=None) -> Dict:
        """
        运行所有检查
        
        Args:
            progress_callback: 进度回调函数 (current, total, message)
            
        Returns:
            检查结果
        """
        total_checks = 8
        current = 0
        
        # 1. 连接检查
        current += 1
        if progress_callback:
            progress_callback(current, total_checks, "检查 HFSS 连接...")
        self._check_connection()
        
        if '连接 HFSS' in self.results['errors'] or not self.hfss:
            return self.results
        
        # 保存原始变量值
        self._save_original_values()
        
        try:
            # 2. 项目文件检查
            current += 1
            if progress_callback:
                progress_callback(current, total_checks, "检查项目文件...")
            self._check_project()
            
            # 3. 变量定义检查
            current += 1
            if progress_callback:
                progress_callback(current, total_checks, "检查变量定义...")
            self._check_variables()
            
            # 4. 变量边界测试
            current += 1
            if progress_callback:
                progress_callback(current, total_checks, "测试变量边界...")
            self._check_variable_bounds()
            
            # 5. 随机采样测试
            current += 1
            if progress_callback:
                progress_callback(current, total_checks, f"随机采样测试 ({self.n_samples} 组)...")
            self._check_random_samples(n_samples=self.n_samples)
            
            # 6. Setup 检查
            current += 1
            if progress_callback:
                progress_callback(current, total_checks, "检查 Setup 配置...")
            self._check_setup()
            
            # 7. 远场设置检查
            current += 1
            if progress_callback:
                progress_callback(current, total_checks, "检查远场设置...")
            self._check_far_field_setup()
            
            # 8. 目标频率检查
            current += 1
            if progress_callback:
                progress_callback(current, total_checks, "检查目标频率...")
            self._check_target_frequencies()
            
        finally:
            # 恢复原始变量值
            self._restore_original_values()
        
        # 汇总
        self._summarize()
        
        return self.results
    
    def _save_original_values(self):
        """保存原始变量值"""
        if not self.hfss or not self.hfss._connected:
            return
        
        try:
            variables = self.config.get('variables', [])
            for var in variables:
                name = var.get('name', '')
                if name:
                    try:
                        value = self.hfss.hfss[name]
                        self.original_values[name] = value
                    except:
                        pass
        except Exception as e:
            print(f"保存原始变量值失败: {e}")
    
    def _restore_original_values(self):
        """恢复原始变量值"""
        if not self.hfss or not self.hfss._connected:
            return
        
        try:
            for name, value in self.original_values.items():
                try:
                    self.hfss.hfss[name] = value
                except:
                    pass
            print("已恢复原始变量值")
        except Exception as e:
            print(f"恢复原始变量值失败: {e}")
    
    def _check_connection(self):
        """检查 HFSS 连接"""
        try:
            from core.hfss_controller import HFSSController
            
            hfss_config = self.config.get('hfss', {})
            
            if not hfss_config.get('project_path'):
                self.results['errors'].append("项目路径未设置")
                return
            
            if not os.path.exists(hfss_config['project_path']):
                self.results['errors'].append(f"项目文件不存在: {hfss_config['project_path']}")
                return
            
            self.hfss = HFSSController(
                hfss_config.get('project_path', ''),
                hfss_config.get('design_name', ''),
                hfss_config.get('setup_name', 'Setup1'),
                hfss_config.get('sweep_name', 'Sweep')
            )
            
            if self.hfss.connect():
                self.results['passed'].append("HFSS 连接成功")
                self.results['details']['connection'] = {
                    'project': hfss_config.get('project_path', ''),
                    'design': hfss_config.get('design_name', ''),
                    'status': 'connected'
                }
            else:
                self.results['errors'].append("HFSS 连接失败")
                
        except Exception as e:
            self.results['errors'].append(f"连接检查异常: {str(e)}")
    
    def _check_project(self):
        """检查项目文件"""
        if not self.hfss or not self.hfss._connected:
            self.results['errors'].append("HFSS 未连接，跳过项目检查")
            return
        
        try:
            # 检查设计
            design_name = self.config['hfss'].get('design_name', '')
            if design_name:
                designs = self.hfss.hfss.design_list
                if design_name not in designs:
                    self.results['errors'].append(f"设计 '{design_name}' 不存在")
                    self.results['details']['available_designs'] = designs
                else:
                    self.results['passed'].append(f"设计 '{design_name}' 存在")
            
            # 检查 Setup
            setup_name = self.config['hfss'].get('setup_name', 'Setup1')
            setups = [s.name for s in self.hfss.hfss.setups]
            if setups:
                if setup_name in setups:
                    self.results['passed'].append(f"Setup '{setup_name}' 存在")
                else:
                    self.results['warnings'].append(f"Setup '{setup_name}' 不存在，将创建")
                    self.results['details']['available_setups'] = setups
            else:
                self.results['warnings'].append("没有 Setup，优化时会自动创建")
                
        except Exception as e:
            self.results['errors'].append(f"项目检查异常: {str(e)}")
    
    def _check_variables(self):
        """检查变量定义"""
        variables = self.config.get('variables', [])
        
        if not variables:
            self.results['errors'].append("没有定义优化变量")
            return
        
        # 检查每个变量
        hfss_vars = []
        try:
            if self.hfss and self.hfss._connected:
                hfss_vars = list(self.hfss.hfss.variable_manager.variable_names)
        except Exception as e:
            print(f"获取 HFSS 变量列表失败: {e}")
        
        missing_vars = []
        valid_vars = []
        
        for var in variables:
            name = var.get('name', '')
            bounds = var.get('bounds', (0, 1))
            
            # 检查变量名
            if not name:
                self.results['errors'].append("有变量未设置名称")
                continue
            
            # 检查边界
            if len(bounds) != 2:
                self.results['errors'].append(f"变量 '{name}' 边界设置错误: {bounds}")
                continue
            
            if bounds[0] >= bounds[1]:
                self.results['errors'].append(f"变量 '{name}' 边界无效: min={bounds[0]}, max={bounds[1]}")
                continue
            
            # 检查变量是否在 HFSS 中定义
            if hfss_vars and name not in hfss_vars:
                missing_vars.append(name)
            else:
                valid_vars.append(name)
        
        if missing_vars:
            self.results['warnings'].append(f"以下变量在 HFSS 中未定义: {missing_vars}")
            self.results['details']['missing_variables'] = missing_vars
        
        if valid_vars:
            self.results['passed'].append(f"{len(valid_vars)} 个变量定义正确: {valid_vars}")
        
        self.results['details']['variables'] = [v.get('name') for v in variables]
    
    def _get_hfss_messages(self, debug: bool = True) -> List[str]:
        """获取 HFSS 消息日志"""
        messages = []
        try:
            if self.hfss and self.hfss._connected:
                # 方法1: 通过 odesign 获取消息
                try:
                    odesign = self.hfss.hfss._odesign
                    
                    if debug:
                        print(f"  [DEBUG] odesign 类型: {type(odesign).__name__}")
                    
                    # 尝试 ClearMessages 后的消息计数方法
                    if hasattr(odesign, 'GetMessages'):
                        msgs = odesign.GetMessages()
                        
                        if debug:
                            print(f"  [DEBUG] GetMessages 返回类型: {type(msgs).__name__}")
                        
                        # COM 对象可能需要特殊处理
                        try:
                            # 方法 A: 尝试作为迭代器
                            count = 0
                            for msg in msgs:
                                messages.append(str(msg))
                                count += 1
                            if debug:
                                print(f"  [DEBUG] 迭代获取到 {count} 条消息")
                        except TypeError:
                            # 方法 B: 尝试索引访问
                            try:
                                for i in range(100):
                                    try:
                                        msg = msgs[i]
                                        messages.append(str(msg))
                                    except IndexError:
                                        break
                                if debug:
                                    print(f"  [DEBUG] 索引获取到 {len(messages)} 条消息")
                            except:
                                pass
                        
                        # 方法 C: 尝试 Count 和 Item 属性
                        if not messages:
                            try:
                                if hasattr(msgs, 'Count'):
                                    count = msgs.Count
                                    if debug:
                                        print(f"  [DEBUG] msgs.Count = {count}")
                                    for i in range(count):
                                        try:
                                            if hasattr(msgs, 'Item'):
                                                msg = msgs.Item(i)
                                            else:
                                                msg = msgs[i]
                                            messages.append(str(msg))
                                        except:
                                            pass
                            except Exception as e:
                                if debug:
                                    print(f"  [DEBUG] Count/Item 失败: {e}")
                        
                        # 方法 D: 转换为字符串列表
                        if not messages:
                            try:
                                messages = [str(m) for m in list(msgs)]
                                if debug:
                                    print(f"  [DEBUG] list() 转换获取到 {len(messages)} 条消息")
                            except:
                                pass
                    
                    else:
                        if debug:
                            print("  [DEBUG] odesign 没有 GetMessages 方法")
                            
                except Exception as e:
                    if debug:
                        print(f"  [DEBUG] GetMessages 异常: {e}")
                
                # 方法2: 通过 desktop 获取
                if not messages:
                    try:
                        desktop = self.hfss.hfss._desktop
                        if hasattr(desktop, 'GetMessages'):
                            msgs = desktop.GetMessages()
                            if debug:
                                print(f"  [DEBUG] desktop.GetMessages 返回类型: {type(msgs).__name__}")
                            try:
                                messages = [str(m) for m in msgs]
                            except:
                                pass
                    except Exception as e:
                        if debug:
                            print(f"  [DEBUG] desktop 获取失败: {e}")
                        
        except Exception as e:
            if debug:
                print(f"  [DEBUG] 获取消息总异常: {e}")
        
        if debug and messages:
            print(f"  [DEBUG] 最终获取到 {len(messages)} 条消息")
            for i, msg in enumerate(messages[-5:]):
                print(f"  [DEBUG] 消息[{i}]: {msg[:80]}...")
        
        return messages
    
    def _clear_messages(self):
        """清除 HFSS 消息"""
        try:
            if self.hfss and self.hfss._connected:
                try:
                    self.hfss.hfss._odesign.ClearMessages()
                except:
                    pass
        except:
            pass
    
    def _validate_design(self) -> Tuple[bool, List[str]]:
        """
        使用 HFSS Validate 功能验证设计
        
        Returns:
            (is_valid, error_messages)
        """
        errors = []
        
        try:
            if not self.hfss or not self.hfss._connected:
                return False, ["HFSS 未连接"]
            
            # 方法1: 使用 PyAEDT 内置的 validate
            try:
                if hasattr(self.hfss.hfss, 'validate'):
                    result = self.hfss.hfss.validate()
                    # validate 通常返回 True/False 或验证结果对象
                    if result is False:
                        errors.append("设计验证失败")
                    return result, errors
            except Exception as e:
                print(f"  [DEBUG] hfss.validate() 失败: {e}")
            
            # 方法2: 使用 odesign.ValidateDesign()
            try:
                odesign = self.hfss.hfss._odesign
                if hasattr(odesign, 'ValidateDesign'):
                    result = odesign.ValidateDesign()
                    if result is False or result == 0:
                        errors.append("设计验证失败")
                    return bool(result), errors
            except Exception as e:
                print(f"  [DEBUG] ValidateDesign() 失败: {e}")
            
            # 方法3: 使用 desktop 脚本执行验证
            try:
                # 获取验证结果
                desktop = self.hfss.hfss._desktop
                
                # 尝试运行验证脚本
                if hasattr(desktop, 'RunScript'):
                    # 创建临时脚本
                    import tempfile
                    script_content = '''
Set oProject = oDesktop.GetActiveProject()
Set oDesign = oProject.GetActiveDesign()
oDesign.ValidateDesign
'''
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.vbs', delete=False) as f:
                        f.write(script_content)
                        script_path = f.name
                    
                    try:
                        desktop.RunScript(script_path)
                        # 检查消息获取验证结果
                        time.sleep(0.5)
                    finally:
                        os.unlink(script_path)
                        
            except Exception as e:
                print(f"  [DEBUG] 脚本验证失败: {e}")
            
            # 方法4: 如果以上都失败，尝试通过消息判断
            # (这是最后的备用方案)
            try:
                messages = self._get_hfss_messages(debug=False)
                
                for msg in messages:
                    msg_str = str(msg)
                    msg_lower = msg_str.lower()
                    
                    # 检测错误关键词
                    error_keywords = [
                        'parasolid error', 
                        'pk_error', 
                        'missing_geom',
                        'body could not be created',
                        'invalid parameters',
                        'boolean_2',
                    ]
                    
                    if any(kw in msg_lower for kw in error_keywords):
                        if '[warning]' not in msg_lower:
                            errors.append(msg_str[:150])
                            
            except Exception as e:
                pass
            
            return len(errors) == 0, errors
            
        except Exception as e:
            return False, [f"验证异常: {str(e)[:50]}"]
    
    def _check_model_validity(self, debug: bool = True) -> Tuple[bool, List[str]]:
        """
        检查模型有效性
        
        优先使用 Validate 功能
        
        Returns:
            (is_valid, error_messages)
        """
        # 首先尝试 Validate
        is_valid, errors = self._validate_design()
        
        if debug:
            print(f"  [DEBUG] Validate 结果: {is_valid}, 错误数: {len(errors)}")
        
        return is_valid, errors
    
    def _check_variable_bounds(self):
        """测试变量边界值"""
        if not self.hfss or not self.hfss._connected:
            self.results['warnings'].append("HFSS 未连接，跳过边界测试")
            return
        
        variables = self.config.get('variables', [])
        if not variables:
            return
        
        print("\n[边界测试] 测试变量边界值...")
        print("  使用 HFSS Validate 功能验证...")
        
        failed_bounds = []
        passed_count = 0
        
        for var in variables:
            name = var.get('name', '')
            bounds = var.get('bounds', (0, 1))
            unit = var.get('unit', 'mm')
            
            var_errors = []
            
            # 测试最小值
            try:
                self.hfss.hfss[name] = f"{bounds[0]}{unit}"
                time.sleep(0.5)
                
                # 使用 Validate 验证
                is_valid, errors = self._validate_design()
                if not is_valid:
                    var_errors.append(f"min={bounds[0]}: 验证失败")
                    print(f"  {name}[min={bounds[0]}]: 验证失败 ✗")
                else:
                    print(f"  {name}[min={bounds[0]}]: 通过 ✓")
                
            except Exception as e:
                var_errors.append(f"min={bounds[0]}: 设置失败 {str(e)[:30]}")
                print(f"  {name}[min={bounds[0]}]: 设置失败 ✗")
            
            # 测试最大值
            try:
                self.hfss.hfss[name] = f"{bounds[1]}{unit}"
                time.sleep(0.5)
                
                is_valid, errors = self._validate_design()
                if not is_valid:
                    var_errors.append(f"max={bounds[1]}: 验证失败")
                    print(f"  {name}[max={bounds[1]}]: 验证失败 ✗")
                else:
                    print(f"  {name}[max={bounds[1]}]: 通过 ✓")
                
            except Exception as e:
                var_errors.append(f"max={bounds[1]}: 设置失败 {str(e)[:30]}")
                print(f"  {name}[max={bounds[1]}]: 设置失败 ✗")
            
            if var_errors:
                failed_bounds.append({
                    'name': name,
                    'errors': var_errors
                })
            else:
                passed_count += 1
                print(f"  {name}: 边界测试通过")
        
        if failed_bounds:
            self.results['warnings'].append(
                f"{len(failed_bounds)}/{len(variables)} 个变量边界会导致模型错误"
            )
            self.results['details']['failed_bounds'] = failed_bounds
        else:
            self.results['passed'].append(f"所有 {len(variables)} 个变量边界测试通过")
        
        if passed_count > 0 and failed_bounds:
            self.results['passed'].append(f"{passed_count}/{len(variables)} 个变量边界测试通过")
    
    def _check_random_samples(self, n_samples: int = 10):
        """
        随机采样测试
        
        测试多组随机变量组合，使用 HFSS Validate 检查模型是否有效
        """
        if not self.hfss or not self.hfss._connected:
            self.results['warnings'].append("HFSS 未连接，跳过随机采样测试")
            return
        
        variables = self.config.get('variables', [])
        if not variables:
            return
        
        print(f"\n[随机采样] 测试 {n_samples} 组随机变量组合...")
        print("  使用 HFSS Validate 功能验证...")
        
        failed_samples = []
        success_count = 0
        
        for i in range(n_samples):
            # 生成随机参数
            params = {}
            param_display = {}
            for var in variables:
                name = var.get('name', '')
                bounds = var.get('bounds', (0, 1))
                unit = var.get('unit', 'mm')
                
                # 随机值（在边界内）
                value = random.uniform(bounds[0], bounds[1])
                params[name] = f"{value:.4f}{unit}"
                param_display[name] = f"{value:.2f}{unit}"
            
            # 设置参数
            set_errors = []
            for name, value in params.items():
                try:
                    self.hfss.hfss[name] = value
                except Exception as e:
                    set_errors.append(f"{name}={value}: {str(e)[:30]}")
            
            if set_errors:
                failed_samples.append({
                    'sample': i + 1,
                    'params': param_display,
                    'errors': set_errors,
                    'type': '设置失败'
                })
                print(f"  样本 {i+1}: 参数设置失败")
                continue
            
            # 等待模型更新
            time.sleep(0.5)
            
            # 使用 Validate 验证设计
            is_valid, errors = self._validate_design()
            
            if not is_valid:
                failed_samples.append({
                    'sample': i + 1,
                    'params': param_display,
                    'errors': errors if errors else ['验证失败'],
                    'type': '模型无效'
                })
                print(f"  样本 {i+1}: Validate 失败 ✗")
            else:
                success_count += 1
                print(f"  样本 {i+1}: 通过 ✓")
        
        # 统计结果
        fail_count = len(failed_samples)
        if fail_count > 0:
            fail_rate = fail_count / n_samples * 100
            
            if fail_rate >= 50:
                self.results['errors'].append(
                    f"随机采样失败率过高: {fail_rate:.0f}% ({fail_count}/{n_samples})"
                )
            elif fail_count > 0:
                self.results['warnings'].append(
                    f"随机采样失败: {fail_count}/{n_samples} ({fail_rate:.0f}%)"
                )
            
            # 保存失败样本详情
            if fail_count <= 5:
                # 失败样本少，显示完整参数
                self.results['details']['failed_samples'] = failed_samples
            else:
                # 失败样本多，只保存前5个
                self.results['details']['failed_samples'] = failed_samples[:5]
                self.results['details']['total_failed'] = fail_count
            
            self.results['details']['success_rate'] = f"{100 - fail_rate:.0f}%"
            self.results['failed_params'] = failed_samples
        else:
            self.results['passed'].append(f"随机采样全部通过 ({success_count}/{n_samples})")
    
    def _check_setup(self):
        """检查 Setup 配置"""
        if not self.hfss or not self.hfss._connected:
            return
        
        try:
            setup_name = self.config['hfss'].get('setup_name', 'Setup1')
            setup = self.hfss.hfss.get_setup(setup_name)
            
            if setup:
                # 安全获取属性
                try:
                    freq = setup.props.get('Frequency', 'N/A') if hasattr(setup.props, 'get') else 'N/A'
                except:
                    freq = 'N/A'
                
                print(f"  Setup 频率: {freq}")
                
                # 检查远场球面关联
                try:
                    ff_index = setup.props.get('InfiniteSphereSetup', -1) if hasattr(setup.props, 'get') else -1
                except:
                    ff_index = -1
                
                if ff_index == -1 or ff_index == 0:
                    self.results['warnings'].append("远场球面未关联到 Setup")
                else:
                    self.results['passed'].append(f"远场球面已关联 (索引 {ff_index})")
                
                # 检查 Sweep
                sweeps = []
                try:
                    if hasattr(setup, 'sweeps'):
                        sweeps = [s.name for s in setup.sweeps]
                except:
                    pass
                
                if sweeps:
                    for sweep_name in sweeps:
                        try:
                            sweep = setup.get_sweep(sweep_name)
                            if hasattr(sweep, 'props') and hasattr(sweep.props, 'get'):
                                save_rad = sweep.props.get('SaveRadFields', False)
                                if not save_rad:
                                    self.results['warnings'].append(
                                        f"Sweep '{sweep_name}' 未保存辐射场"
                                    )
                                else:
                                    self.results['passed'].append(f"Sweep '{sweep_name}' 已保存辐射场")
                        except:
                            pass
                
                self.results['details']['setup'] = {
                    'frequency': freq,
                    'far_field_index': ff_index,
                    'sweeps': sweeps
                }
                
        except Exception as e:
            self.results['warnings'].append(f"Setup 检查异常: {str(e)}")
    
    def _check_far_field_setup(self):
        """检查远场设置（辐射边界和远场球体）"""
        if not self.hfss or not self.hfss._connected:
            return
        
        # 检查是否有增益目标
        objectives = self.config.get('objectives', [])
        has_gain_target = any(obj.get('type') == 'peak_gain' for obj in objectives)
        
        if not has_gain_target:
            # 没有增益目标，不需要检查远场设置
            return
        
        print("\n[远场检查] 检查辐射边界和远场球体...")
        
        try:
            # 使用 HFSSController 的检查方法
            status = self.hfss.check_far_field_setup()
            
            # 检查辐射边界
            if status['has_radiation_boundary']:
                self.results['passed'].append("辐射边界 (Radiation Boundary) 已设置")
            else:
                self.results['warnings'].append("未检测到辐射边界 - 如需计算增益请添加")
            
            # 检查远场球体
            if status['has_far_field_sphere']:
                self.results['passed'].append(f"远场球体 '{status['far_field_sphere_name']}' 已创建")
            else:
                self.results['errors'].append("缺少远场球体 (Far Field Sphere) - 无法计算增益！")
                self.results['details']['far_field_help'] = {
                    'instruction': "请在 HFSS 中添加: Radiation -> Insert Far Field Setup -> Infinite Sphere"
                }
            
            # 检查是否关联到 Setup
            if status['is_linked_to_setup']:
                self.results['passed'].append("远场球体已关联到 Setup")
            elif status['has_far_field_sphere']:
                self.results['warnings'].append("远场球体未关联到 Setup - 程序会自动关联")
            
            # 保存详情
            self.results['details']['far_field'] = {
                'has_radiation_boundary': status['has_radiation_boundary'],
                'has_far_field_sphere': status['has_far_field_sphere'],
                'far_field_sphere_name': status['far_field_sphere_name'],
                'is_linked_to_setup': status['is_linked_to_setup'],
                'can_get_gain': status['can_get_gain']
            }
            
            # 如果不能获取增益，提供帮助信息
            if not status['can_get_gain']:
                self.results['errors'].append("远场设置不完整 - 无法计算增益！")
                print("\n" + "=" * 60)
                print("[帮助] 如何在 HFSS 中设置远场:")
                print("  1. 创建辐射边界:")
                print("     - 选择空气腔外表面")
                print("     - HFSS -> Boundaries -> Assign -> Radiation")
                print("  2. 创建远场球体:")
                print("     - HFSS -> Radiation -> Insert Far Field Setup -> Infinite Sphere")
                print("     - 名称设为 '3D'")
                print("     - Theta: 0-180°, 步长 10°")
                print("     - Phi: 0-360°, 步长 10°")
                print("  3. 在 Setup 中关联远场球体:")
                print("     - 打开 Setup -> Advanced -> Far Field Sphere")
                print("     - 选择 '3D'")
                print("=" * 60)
            
        except Exception as e:
            self.results['warnings'].append(f"远场检查异常: {str(e)}")
    
    def _check_target_frequencies(self):
        """检查目标频率设置"""
        objectives = self.config.get('objectives', [])
        
        for obj in objectives:
            obj_type = obj.get('type', '')
            name = obj.get('name', '')
            
            if obj_type == 'peak_gain':
                freq = obj.get('freq')
                if freq:
                    # 检查 Setup 频率
                    if self.hfss and self.hfss._connected:
                        try:
                            setup_freq = self.hfss.get_setup_frequency()
                            if setup_freq and abs(setup_freq - freq) > 0.1:
                                self.results['warnings'].append(
                                    f"目标 '{name}' 频率 {freq}GHz != Setup 频率 {setup_freq}GHz"
                                )
                                self.results['details'][f'{name}_freq_mismatch'] = {
                                    'target': freq,
                                    'setup': setup_freq
                                }
                            else:
                                self.results['passed'].append(f"目标 '{name}' 频率 {freq}GHz 已配置")
                        except Exception as e:
                            self.results['warnings'].append(f"无法检查 Setup 频率: {str(e)[:50]}")
            
            elif obj_type == 's_db':
                freq_range = obj.get('freq_range')
                if freq_range:
                    self.results['passed'].append(f"目标 '{name}' 频率范围 {freq_range}GHz 已配置")
    
    def _summarize(self):
        """汇总结果"""
        total = len(self.results['passed']) + len(self.results['warnings']) + len(self.results['errors'])
        
        # 根据错误和警告数量决定状态
        if self.results['errors']:
            status = 'ERROR'
        elif self.results['warnings'] and len(self.results['warnings']) > 3:
            status = 'WARNING'
        elif self.results['warnings']:
            status = 'OK_WITH_WARNINGS'
        else:
            status = 'OK'
        
        self.results['summary'] = {
            'total_checks': total,
            'passed': len(self.results['passed']),
            'warnings': len(self.results['warnings']),
            'errors': len(self.results['errors']),
            'status': status
        }
    
    def get_report_text(self) -> str:
        """获取文本报告 - 更清晰的格式"""
        lines = []
        summary = self.results.get('summary', {})
        status = summary.get('status', 'UNKNOWN')
        
        # 状态标题
        if status == 'OK':
            status_text = "✅ 自检通过"
        elif status == 'OK_WITH_WARNINGS':
            status_text = "⚠️ 自检通过（有警告）"
        elif status == 'WARNING':
            status_text = "⚠️ 自检发现较多问题"
        else:
            status_text = "❌ 自检发现问题"
        
        lines.append("=" * 70)
        lines.append(f"HFSS 项目自检报告")
        lines.append(f"状态: {status_text}")
        lines.append("=" * 70)
        lines.append("")
        
        # 统计
        lines.append(f"通过: {summary.get('passed', 0)} | 警告: {summary.get('warnings', 0)} | 错误: {summary.get('errors', 0)}")
        lines.append("")
        
        # 错误项（最重要）
        if self.results['errors']:
            lines.append("━" * 70)
            lines.append("❌ 错误项（必须修复）:")
            lines.append("━" * 70)
            for item in self.results['errors']:
                lines.append(f"  • {item}")
            lines.append("")
        
        # 警告项
        if self.results['warnings']:
            lines.append("━" * 70)
            lines.append("⚠️ 警告项（建议检查）:")
            lines.append("━" * 70)
            for item in self.results['warnings']:
                lines.append(f"  • {item}")
            lines.append("")
        
        # 通过项
        if self.results['passed']:
            lines.append("━" * 70)
            lines.append("✅ 通过项:")
            lines.append("━" * 70)
            for item in self.results['passed']:
                lines.append(f"  • {item}")
            lines.append("")
        
        # 失败的参数组合（如果有）
        failed_params = self.results.get('failed_params', [])
        if failed_params:
            lines.append("━" * 70)
            lines.append(f"📋 失败的参数组合列表 (共 {len(failed_params)} 组):")
            lines.append("━" * 70)
            
            for i, fp in enumerate(failed_params[:10]):  # 最多显示10组
                lines.append(f"\n  [{fp['sample']}] 类型: {fp.get('type', '未知')}")
                lines.append(f"  参数值:")
                
                # 分行显示参数
                params = fp.get('params', {})
                param_strs = [f"    {k}={v}" for k, v in params.items()]
                # 每行3个参数
                for j in range(0, len(param_strs), 3):
                    lines.append("  " + ", ".join(param_strs[j:j+3]))
                
                # 错误信息
                errors = fp.get('errors', [])
                if errors:
                    lines.append(f"  错误:")
                    for e in errors[:2]:  # 最多显示2个错误
                        lines.append(f"    - {e}")
            
            if len(failed_params) > 10:
                lines.append(f"\n  ... 还有 {len(failed_params) - 10} 组失败")
            lines.append("")
        
        # 边界失败详情
        failed_bounds = self.results.get('details', {}).get('failed_bounds', [])
        if failed_bounds:
            lines.append("━" * 70)
            lines.append("📋 变量边界问题详情:")
            lines.append("━" * 70)
            for fb in failed_bounds:
                lines.append(f"\n  变量: {fb['name']}")
                for e in fb['errors']:
                    lines.append(f"    • {e}")
            lines.append("")
        
        # 结论
        lines.append("=" * 70)
        if status == 'ERROR':
            lines.append("❌ 建议: 修复错误后再开始优化")
        elif status == 'WARNING':
            lines.append("⚠️ 建议: 检查警告项，部分变量范围可能需要调整")
        elif status == 'OK_WITH_WARNINGS':
            lines.append("⚠️ 建议: 可以开始优化，但建议检查警告项")
        else:
            lines.append("✅ 所有检查通过，可以开始优化")
        lines.append("=" * 70)
        
        return "\n".join(lines)


def run_check(config: Dict, n_samples: int = 10, progress_callback=None) -> Dict:
    """
    运行项目自检
    
    Args:
        config: 配置字典
        n_samples: 随机采样数量
        progress_callback: 进度回调
        
    Returns:
        检查结果
    """
    checker = ProjectChecker(config, n_samples=n_samples)
    return checker.run_all_checks(progress_callback)


if __name__ == "__main__":
    # 测试
    import json
    
    if os.path.exists("user_config.json"):
        with open("user_config.json", 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        def progress(current, total, message):
            print(f"[{current}/{total}] {message}")
        
        checker = ProjectChecker(config, n_samples=10)
        results = checker.run_all_checks(progress)
        
        print("\n" + checker.get_report_text())
    else:
        print("未找到 user_config.json")
