# % 二体轨道快速计算
# % 输入：coe0 - 初始轨道根数      T - 递推时间
# % 输出：coe - 最终轨道根数
import math
from OrbitCore.mean2True import mean2True
from OrbitCore.true2Mean  import true2Mean
from OrbitCore.amendDeg import amendDeg
import globals
def twoBodyOrbit(coe0, T):
    q = true2Mean(coe0[1], coe0[5])
    Tperiod = 2 * math.pi * math.sqrt(coe0[0] ** 3 / globals.GM_Earth)
    M = 2 * math.pi / Tperiod * T * globals.rad2deg
    q = amendDeg(mean2True(coe0[1], amendDeg(q + M, '0 - 360')), '0 - 360')
    coe = list(coe0)
    coe[5] = q
    return coe
