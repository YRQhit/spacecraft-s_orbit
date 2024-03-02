import numpy as np
from ToolFunction.Inertial2Orbit import Inertial2Orbit
import globals
#  RVm 前六项为RV 后面附加质量 考虑J2摄动
#  Thrust_f 为推力大小 推力大小恒定
#  k 为质量变化率 一般为负数

def J2Cal_rvm(t, RVm, Thrust_f, deg, k):
    GM_Earth, J2, r_E = globals.GM_Earth, globals.J2, globals.r_E
    Azimuth = deg[0]
    Elevation = deg[1]
    x = RVm[3]
    y = RVm[4]
    z = RVm[5]
    m = RVm[6]

    # 脉冲轨道系定向，需要求解对应时刻惯性系的脉冲
    RotMat = np.transpose(Inertial2Orbit(RVm))
    vec = np.array([np.cos(np.radians(Elevation)) * np.cos(np.radians(Azimuth)),
                    np.cos(np.radians(Elevation)) * np.sin(np.radians(Azimuth)),
                    np.sin(np.radians(Elevation))])
    accel = Thrust_f / m / 1000  # 根据推力计算加速度大小
    r = np.linalg.norm(RVm[0:3])
    dx = accel * np.dot(RotMat[0], vec) - GM_Earth * RVm[0] / r ** 3 * (1 + 1.5 * J2 * (r_E / r) ** 2 * (1 - 5 * RVm[2] ** 2 / r ** 2))
    dy = accel * np.dot(RotMat[1], vec) - GM_Earth * RVm[0] / r ** 3 * (1 + 1.5 * J2 * (r_E / r) ** 2 * (1 - 5 * RVm[2] ** 2 / r ** 2)) * (RVm[1] / RVm[0])
    dz = accel * np.dot(RotMat[2], vec) - GM_Earth * RVm[2] / r ** 3 * (1 + 1.5 * J2 * (r_E / r) ** 2 * (3 - 5 * RVm[2] ** 2 / r ** 2))

    drvm = [x, y, z, dx, dy, dz, k]
    return drvm
