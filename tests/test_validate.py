"""
测试 PyAEDT 的 Validate 功能

HFSS 的 Validate 功能可以检查模型是否有效
"""
import sys
import os
import time

PROJECT_PATH = r"C:\Users\16438\Desktop\pyHFSSProject\n2.aedt"
DESIGN_NAME = "S1"

def test_validate():
    """测试 Validate 功能"""
    print("=" * 60)
    print("测试 HFSS Validate 功能")
    print("=" * 60)
    
    try:
        import pyaedt
        print(f"PyAEDT 版本: {pyaedt.__version__}")
    except ImportError:
        print("无法导入 pyaedt")
        return
    
    # 连接到 HFSS
    print(f"\n连接到项目: {PROJECT_PATH}")
    
    try:
        hfss = pyaedt.Hfss(
            projectname=PROJECT_PATH,
            designname=DESIGN_NAME,
            solution_type="Terminal",
            specified_version="2023.1",
            new_desktop_session=False,
            close_on_exit=False
        )
    except Exception as e:
        print(f"连接失败: {e}")
        return
    
    print("✓ 连接成功")
    
    # 查找 validate 相关方法
    print("\n" + "=" * 60)
    print("查找 Validate 方法:")
    print("=" * 60)
    
    # 在 hfss 对象上查找
    validate_methods = [m for m in dir(hfss) if 'valid' in m.lower()]
    print(f"\nhfss 对象的 validate 相关方法: {validate_methods}")
    
    # 在 modeler 上查找
    modeler_validate = [m for m in dir(hfss.modeler) if 'valid' in m.lower()]
    print(f"modeler 对象的 validate 相关方法: {modeler_validate}")
    
    # 在 odesign 上查找
    try:
        odesign = hfss._odesign
        odesign_methods = [m for m in dir(odesign) if 'valid' in m.lower()]
        print(f"odesign 对象的 validate 相关方法: {odesign_methods}")
    except:
        pass
    
    # 测试常见的方法名
    print("\n" + "=" * 60)
    print("测试方法:")
    print("=" * 60)
    
    # 方法1: hfss.validate()
    print("\n[方法1] hfss.validate()")
    if hasattr(hfss, 'validate'):
        try:
            result = hfss.validate()
            print(f"  返回值: {result}, 类型: {type(result)}")
        except Exception as e:
            print(f"  错误: {e}")
    else:
        print("  不存在此方法")
    
    # 方法2: modeler.validate()
    print("\n[方法2] modeler.validate()")
    if hasattr(hfss.modeler, 'validate'):
        try:
            result = hfss.modeler.validate()
            print(f"  返回值: {result}, 类型: {type(result)}")
        except Exception as e:
            print(f"  错误: {e}")
    else:
        print("  不存在此方法")
    
    # 方法3: odesign.ValidateDesign()
    print("\n[方法3] odesign.ValidateDesign()")
    try:
        odesign = hfss._odesign
        if hasattr(odesign, 'ValidateDesign'):
            result = odesign.ValidateDesign()
            print(f"  返回值: {result}, 类型: {type(result)}")
        elif hasattr(odesign, 'Validate'):
            result = odesign.Validate()
            print(f"  返回值: {result}, 类型: {type(result)}")
        else:
            print("  不存在此方法")
            # 列出所有方法
            all_methods = [m for m in dir(odesign) if not m.startswith('_') and not m.startswith('Get')]
            print(f"  可用方法: {all_methods[:20]}...")
    except Exception as e:
        print(f"  错误: {e}")
    
    # 方法4: 通过脚本执行验证
    print("\n[方法4] 执行验证脚本")
    try:
        # HFSS 脚本方式
        script = """
Dim oProject
Set oProject = oDesktop.GetActiveProject()
Dim oDesign
Set oDesign = oProject.GetActiveDesign()
oDesign.ValidateDesign
"""
        # 查找执行脚本的方法
        if hasattr(hfss, '_desktop'):
            desktop = hfss._desktop
            if hasattr(desktop, 'RunScript'):
                print("  找到 RunScript 方法")
    except Exception as e:
        print(f"  错误: {e}")
    
    # 方法5: 测试变量修改后的 validate
    print("\n" + "=" * 60)
    print("测试修改变量后 Validate:")
    print("=" * 60)
    
    # 保存原始值
    test_var = "Rl"
    try:
        original_value = hfss[test_var]
        print(f"\n原始值: {test_var} = {original_value}")
        
        # 修改为一个可能有效的值
        hfss[test_var] = "20mm"
        time.sleep(0.5)
        print(f"修改后: {test_var} = 20mm")
        
        # 尝试 validate
        if hasattr(hfss, 'validate'):
            result = hfss.validate()
            print(f"Validate 结果: {result}")
        
        # 检查消息
        if hasattr(hfss._odesign, 'GetMessages'):
            msgs = hfss._odesign.GetMessages()
            print(f"消息数量: {len(list(msgs)) if hasattr(msgs, '__len__') else 'N/A'}")
        
        # 恢复原值
        hfss[test_var] = original_value
        
    except Exception as e:
        print(f"测试错误: {e}")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    test_validate()
