#  RVm 前六项为RV 后面附加质量
#  Thrust_f 为推力大小 推力大小恒定
#  k 为质量变化率 一般为负数
import numpy as np
from ToolFunction.Inertial2Orbit import Inertial2Orbit
import globals
def TwoBodyCal(t, RV, Thrust_f, deg):
    GM_Earth = globals.GM_Earth
    Azimuth = deg[0]
    Elevation = deg[1]

    x = RV[3]
    y = RV[4]
    z = RV[5]

    # 脉冲轨道系定向  需要求解对应时刻惯性系的脉冲
    RotMat = Inertial2Orbit(RV).T
    vec = [np.cos(np.radians(Elevation)) * np.cos(np.radians(Azimuth)),
           np.cos(np.radians(Elevation)) * np.sin(np.radians(Azimuth)),
           np.sin(np.radians(Elevation))]

    r = np.linalg.norm(RV[:3])
    dx = -GM_Earth / r ** 3 * RV[0] + Thrust_f * np.dot(RotMat[0], vec)
    dy = -GM_Earth / r ** 3 * RV[1] + Thrust_f * np.dot(RotMat[1], vec)
    dz = -GM_Earth / r ** 3 * RV[2] + Thrust_f * np.dot(RotMat[2], vec)

    drv = np.array([x, y, z, dx, dy, dz])
    return drv
