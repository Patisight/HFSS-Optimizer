"""
测试 S 参数获取功能
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.hfss_controller import HFSSController
import numpy as np

# 项目配置
project_path = "C:/Users/16438/Desktop/pyHFSSProject/n2.aedt"
design_name = "S3"
setup_name = "Setup1"
sweep_name = "Sweep"

# 测试参数
test_params = [
    1.274019740012153,
    0.8906459826447346,
    0.6906946393787124,
    2.2666489580041738,
    12.227820622159442,
    12.70619423017497,
    1.8285791590519929,
    0.5,
    1.0601403252636028,
    1.523385747266244,
    4.256448603130282,
    2.4414378850789507,
    4.416325040266694
]
var_names = ['Wm', 'Wm2', 'Wm3', 'Md', 'Wp', 'Lp', 'Lp2', 'Sangle', 'Wm4', 'Wd', 'Lm', 'Lm4', 'Wc']

print("=" * 60)
print("测试 S 参数获取")
print("=" * 60)

# 创建 HFSS 控制器
hfss = HFSSController(project_path, design_name, setup_name, sweep_name)

# 连接
if not hfss.connect():
    print("[ERROR] 无法连接到 HFSS")
    sys.exit(1)
print("[OK] 已连接到 HFSS")

# 设置变量
print("\n[INFO] 设置变量...")
for name, value in zip(var_names, test_params):
    hfss.set_variable(name, value)
print("[OK] 变量设置完成")

# 仿真
print("\n[INFO] 开始仿真...")
if not hfss.analyze(force=True):
    print("[ERROR] 仿真失败")
    hfss.close()
    sys.exit(1)
print("[OK] 仿真完成")

# 获取 S 参数 - 测试多个端口
print("\n[INFO] 测试获取 S(1,1)...")
s_data = hfss.get_s_parameters([(1, 1)])
if s_data is None:
    print("[ERROR] 获取 S(1,1) 失败")
    hfss.close()
    sys.exit(1)

print(f"[OK] 获取到 {len(s_data['freq'])} 个频率点")
print(f"  频率范围: {s_data['freq'].min():.4f} - {s_data['freq'].max():.4f} GHz")

# 检查 S(1,1) 数据
port_key = (1, 1)
if port_key in s_data['ports']:
    port_data = s_data['ports'][port_key]
    print(f"\nS(1,1) 数据:")
    print(f"  keys: {port_data.keys()}")
    print(f"  db shape: {port_data['db'].shape if 'db' in port_data else 'N/A'}")
    print(f"  real shape: {port_data['real'].shape if 'real' in port_data else 'N/A'}")
    print(f"  imag shape: {port_data['imag'].shape if 'imag' in port_data else 'N/A'}")
    
    if 'db' in port_data:
        s11_db = port_data['db']
        print(f"  dB range: {s11_db.min():.2f} to {s11_db.max():.2f} dB")
    
    # 测试从 dB 估算幅值的情况
    if 'real' in port_data and port_data['real'] is not None:
        print(f"  real range: {port_data['real'].min():.4f} to {port_data['real'].max():.4f}")
    else:
        print(f"  real: None (从 dB 估算)")
    
    if 'imag' in port_data and port_data['imag'] is not None:
        print(f"  imag range: {port_data['imag'].min():.4f} to {port_data['imag'].max():.4f}")
    else:
        print(f"  imag: None (从 dB 估算)")

# 测试频段筛选
print("\n" + "=" * 60)
print("测试频段筛选 (5.6 - 6.2 GHz)")
print("=" * 60)

freq = s_data['freq']
mask = (freq >= 5.6) & (freq <= 6.2)
freq_range = freq[mask]
print(f"筛选后频率点数: {len(freq_range)}")

if port_key in s_data['ports'] and 'db' in s_data['ports'][port_key]:
    s11_range = s_data['ports'][port_key]['db'][mask]
    print(f"S(1,1) dB 在 5.6-6.2 GHz:")
    print(f"  min: {s11_range.min():.4f} dB")
    print(f"  max: {s11_range.max():.4f} dB")
    print(f"  mean: {s11_range.mean():.4f} dB")

# 断开连接
hfss.close()
print("\n[OK] 测试完成")
