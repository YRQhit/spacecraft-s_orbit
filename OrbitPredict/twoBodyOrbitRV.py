# 二体轨道快速计算
# 输入：rv0 - 初始轨道位置速度      T - 递推时间
# 输出：rv - 最终轨道位置速度
import numpy as np
from OrbitCore.Orbit_Element_2_State_rv import Orbit_Element_2_State_rv
from OrbitCore.State_rv_2_Orbit_Element import State_rv_2_Orbit_Element
from OrbitPredict.twoBodyOrbit import twoBodyOrbit
def twoBodyOrbitRV(rv0, T):
    # print("调用ing")
    coe0 = State_rv_2_Orbit_Element(rv0[:3], rv0[3:6])
    coe = twoBodyOrbit(coe0, T)
    r, v = Orbit_Element_2_State_rv(coe)
    rv = np.concatenate((r, v))
    return rv
