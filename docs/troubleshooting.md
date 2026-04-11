# 常见问题排查

## HFSS连接问题
### 问题1：无法连接到HFSS
**现象**：日志显示`[ERROR] All connection methods failed`
**解决方法**：
1. 确保HFSS已经正确安装并激活
2. 手动打开HFSS项目，检查是否有错误提示
3. 关闭所有HFSS进程，删除项目目录下的`.lck`锁文件
4. 以管理员身份运行命令行

### 问题2：HFSS连接不稳定，经常断开
**现象**：优化过程中频繁出现重连提示
**解决方法**：
1. 升级pyaedt到最新版本：`pip install --upgrade pyaedt`
2. 减少同时运行的其他程序，释放内存
3. 在配置中增加`hfss.reconnect_attempts = 5`

## 仿真问题
### 问题1：仿真失败，提示"Solution data not available"
**现象**：每次仿真都失败，提示没有求解数据
**解决方法**：
1. 手动在HFSS中运行一次仿真，确保可以正常完成
2. 检查Setup配置是否正确，频率范围是否合理
3. 增加仿真超时时间：`hfss.analysis_timeout = 3600`（单位秒）

### 问题2：无法获取增益数据
**现象**：日志显示`[ERROR] No far field setup found`
**解决方法**：
1. 在HFSS中添加辐射边界：Draw -> Create Radiation Boundary
2. 创建远场球体：Radiation -> Insert Far Field Setup -> Infinite Sphere
3. 在Setup的Advanced选项中关联远场球体

## 优化问题
### 问题1：优化结果不收敛
**现象**：迭代多次后目标值没有明显改善
**解决方法**：
1. 增加种群大小和迭代次数
2. 检查变量范围是否合理，是否包含最优解所在区域
3. 检查目标定义是否合理，是否有冲突
4. 尝试使用其他优化算法

### 问题2：代理模型精度低
**现象**：代理模型预测值和真实仿真值差距大
**解决方法**：
1. 增加初始采样数量（至少是变量数量的5倍）
2. 检查变量范围是否过大，是否有无效区域
3. 尝试使用不同的代理模型（如随机森林、神经网络）
4. 增加迭代次数，让模型有更多数据学习

## 其他问题
### 问题1：图形界面无法启动
**现象**：运行`launch_gui.py`报错
**解决方法**：
1. 安装PyQt6依赖：`pip install PyQt6`
2. 升级pip：`python -m pip install --upgrade pip`
3. 检查Python版本是否为3.8~3.11

### 问题2：打包exe后运行报错
**现象**：双击exe后闪退或提示错误
**解决方法**：
1. 确保打包时使用的Python版本和运行环境一致
2. 检查spec文件中的datas配置是否包含所有需要的资源
3. 尝试以管理员身份运行exe
4. 查看日志文件`logs/`中的错误信息
