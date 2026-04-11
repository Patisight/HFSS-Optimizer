"""
测试 PyAEDT 消息获取方法

运行此脚本来研究如何正确获取 HFSS Message Manager 的内容
"""
import sys
import os
import time

# 项目路径
PROJECT_PATH = r"C:\Users\16438\Desktop\pyHFSSProject\n2.aedt"
DESIGN_NAME = "S1"

def test_message_api():
    """测试不同的消息获取方法"""
    print("=" * 60)
    print("测试 PyAEDT 消息获取 API")
    print("=" * 60)
    
    try:
        import pyaedt
        print(f"\nPyAEDT 版本: {pyaedt.__version__}")
        PYAEDT_NEW_API = False
    except ImportError:
        # 新版 pyaedt (0.25+) 使用新的导入方式
        try:
            from ansys.aedt.core import Hfss as _Hfss
            from ansys.aedt.core import __version__
            print(f"\nPyAEDT 版本: {__version__} [new API]")
            import ansys.aedt.core as pyaedt
            PYAEDT_NEW_API = True
        except ImportError:
            print("无法导入 pyaedt")
            return
    
    # 连接到 HFSS
    print(f"\n连接到项目: {PROJECT_PATH}")
    print(f"设计: {DESIGN_NAME}")
    
    try:
        if PYAEDT_NEW_API:
            hfss = pyaedt.Hfss(
                project=PROJECT_PATH,
                design=DESIGN_NAME,
                solution_type="Terminal",
                new_desktop=True,
            )
        else:
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
    
    print("\n✓ 连接成功")
    
    # 测试各种消息获取方法
    print("\n" + "=" * 60)
    print("方法测试:")
    print("=" * 60)
    
    # 方法 1: _odesign.GetMessages()
    print("\n[方法 1] hfss._odesign.GetMessages()")
    try:
        odesign = hfss._odesign
        print(f"  odesign 类型: {type(odesign)}")
        
        # 检查可用方法
        msg_methods = [m for m in dir(odesign) if 'message' in m.lower() or 'mess' in m.lower()]
        print(f"  消息相关方法: {msg_methods}")
        
        # 尝试获取消息
        if hasattr(odesign, 'GetMessages'):
            messages = odesign.GetMessages()
            print(f"  ✓ GetMessages() 返回类型: {type(messages)}")
            print(f"  ✓ 消息数量: {len(messages) if hasattr(messages, '__len__') else 'N/A'}")
            if messages:
                print(f"  前3条消息:")
                for i, msg in enumerate(list(messages)[:3]):
                    print(f"    [{i}] {str(msg)[:100]}")
        else:
            print("  ✗ 没有 GetMessages 方法")
    except Exception as e:
        print(f"  ✗ 错误: {e}")
    
    # 方法 2: _desktop 获取消息
    print("\n[方法 2] 通过 _desktop 获取消息")
    try:
        desktop = hfss._desktop
        print(f"  desktop 类型: {type(desktop)}")
        
        # 检查可用方法
        msg_methods = [m for m in dir(desktop) if 'message' in m.lower() or 'mess' in m.lower()]
        print(f"  消息相关方法: {msg_methods}")
        
        # 尝试不同的方法
        for method_name in ['GetMessages', 'GetProjectMessages', 'messages']:
            if hasattr(desktop, method_name):
                try:
                    method = getattr(desktop, method_name)
                    result = method()
                    print(f"  ✓ {method_name}() 返回: {type(result)}")
                    if result:
                        print(f"    内容: {str(result)[:200]}")
                except Exception as e:
                    print(f"  ✗ {method_name}() 错误: {e}")
    except Exception as e:
        print(f"  ✗ 错误: {e}")
    
    # 方法 3: 检查 oProject
    print("\n[方法 3] 通过 oProject 获取消息")
    try:
        oproject = hfss.oproject
        print(f"  oproject 类型: {type(oproject)}")
        
        msg_methods = [m for m in dir(oproject) if 'message' in m.lower() or 'mess' in m.lower()]
        print(f"  消息相关方法: {msg_methods}")
        
        # 尝试获取消息
        for method_name in ['GetMessages', 'GetChildObject']:
            if hasattr(oproject, method_name):
                try:
                    method = getattr(oproject, method_name)
                    if method_name == 'GetChildObject':
                        # 尝试获取消息对象
                        try:
                            msg_obj = oproject.GetChildObject('Messages')
                            print(f"  ✓ GetChildObject('Messages'): {type(msg_obj)}")
                            if msg_obj:
                                # 获取消息内容
                                msgs = msg_obj.GetChildNames() if hasattr(msg_obj, 'GetChildNames') else []
                                print(f"    消息数量: {len(msgs)}")
                        except:
                            pass
                    else:
                        result = method()
                        print(f"  ✓ {method_name}() 返回: {type(result)}")
                except Exception as e:
                    print(f"  ✗ {method_name}() 错误: {e}")
    except Exception as e:
        print(f"  ✗ 错误: {e}")
    
    # 方法 4: 使用 Message Manager 对象
    print("\n[方法 4] 查找 Message Manager 对象")
    try:
        # 查找所有包含 message 的属性
        for attr_name in dir(hfss):
            if 'message' in attr_name.lower():
                print(f"  hfss.{attr_name}")
                try:
                    attr = getattr(hfss, attr_name)
                    print(f"    类型: {type(attr)}")
                    if callable(attr):
                        try:
                            result = attr()
                            print(f"    调用结果: {str(result)[:100]}")
                        except:
                            pass
                except:
                    pass
    except Exception as e:
        print(f"  ✗ 错误: {e}")
    
    # 方法 5: 尝试通过 COM API
    print("\n[方法 5] 尝试 COM API")
    try:
        # 获取 MessageManager
        if hasattr(hfss, '_messenger') or hasattr(hfss, 'messagelogger'):
            mm = getattr(hfss, '_messenger', None) or getattr(hfss, 'messagelogger', None)
            print(f"  找到消息管理器: {type(mm)}")
            if mm:
                for m in dir(mm):
                    if not m.startswith('_'):
                        print(f"    {m}")
    except Exception as e:
        print(f"  ✗ 错误: {e}")
    
    # 方法 6: 尝试清除和获取消息
    print("\n[方法 6] 清除消息后测试")
    try:
        # 清除消息
        if hasattr(hfss._odesign, 'ClearMessages'):
            hfss._odesign.ClearMessages()
            print("  ✓ 已清除消息")
        
        # 修改变量触发错误
        print("\n  修改变量触发可能的错误...")
        test_var = "Rl"
        try:
            original = hfss[test_var]
            hfss[test_var] = "1mm"  # 可能触发错误的值
            time.sleep(1)
            
            # 再次获取消息
            if hasattr(hfss._odesign, 'GetMessages'):
                messages = hfss._odesign.GetMessages()
                print(f"  修改后消息数量: {len(messages) if hasattr(messages, '__len__') else 'N/A'}")
                if messages:
                    for msg in list(messages)[:5]:
                        print(f"    {str(msg)[:100]}")
            
            # 恢复原值
            hfss[test_var] = original
        except Exception as e:
            print(f"  变量修改错误: {e}")
            
    except Exception as e:
        print(f"  ✗ 错误: {e}")
    
    # 方法 7: 检查 PyAEDT 的 messages 属性
    print("\n[方法 7] 检查 PyAEDT 内置消息功能")
    try:
        # PyAEDT 可能有自己的消息记录
        if hasattr(hfss, 'logger'):
            logger = hfss.logger
            print(f"  logger 类型: {type(logger)}")
            logger_methods = [m for m in dir(logger) if not m.startswith('_')]
            print(f"  logger 方法: {logger_methods[:10]}")
    except Exception as e:
        print(f"  ✗ 错误: {e}")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
    
    # 不断开连接，保持 HFSS 运行
    # hfss.release_desktop(False, False)


if __name__ == "__main__":
    test_message_api()
