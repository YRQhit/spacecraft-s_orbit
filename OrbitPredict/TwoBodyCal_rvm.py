#  RVm 前六项为RV 后面附加质量
#  Thrust_f 为推力大小 推力大小恒定
#  k 为质量变化率 一般为负数
import numpy as np
from ToolFunction.Inertial2Orbit import Inertial2Orbit
import globals
def TwoBodyCal_rvm(t, RVm, Thrust_f, deg, k):
    GM_Earth = globals.GM_Earth
    Azimuth = deg[0]
    Elevation = deg[1]

    x = RVm[3]
    y = RVm[4]
    z = RVm[5]
    m = RVm[6]

    # 脉冲轨道系定向  需要求解对应时刻惯性系的脉冲
    RotMat = Inertial2Orbit(RVm).T
    vec = [np.cos(np.radians(Elevation)) * np.cos(np.radians(Azimuth)),
           np.cos(np.radians(Elevation)) * np.sin(np.radians(Azimuth)),
           np.sin(np.radians(Elevation))]
    accel = Thrust_f / m / 1000  # 根据推力计算加速度大小
    r = np.linalg.norm(RVm[:3])
    dx = -GM_Earth / r ** 3 * RVm[0] + accel * np.dot(RotMat[0], vec)
    dy = -GM_Earth / r ** 3 * RVm[1] + accel * np.dot(RotMat[1], vec)
    dz = -GM_Earth / r ** 3 * RVm[2] + accel * np.dot(RotMat[2], vec)

    drvm = np.array([x, y, z, dx, dy, dz, k])
    return drvm
