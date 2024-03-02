import numpy as np
import globals
# % 计算大气摄动加速度
# % 输入参数：
# % PosVel - 卫星位置速度       dens - 大气密度（不给定时采用指数模型计算）
# % 输出参数：
# % a - 大气摄动产生的三轴加速度
def AccelDrag(PosVel, dens=None):
    r = np.sqrt(PosVel[0] ** 2 + PosVel[1] ** 2 + PosVel[2] ** 2)

    if dens is None:
        dens = ComputeDensity(r)

    w = 0.7292e-4
    v_rel = np.array([PosVel[3] + w * PosVel[1],
                      PosVel[4] - w * PosVel[0],
                      PosVel[5]])

    a = -0.5 * globals.CD * globals.s_m * 1e3 * dens * np.linalg.norm(v_rel) * v_rel
    return a

# %% 计算大气密度（指数模型）
# % 输入：r - 卫星地心距   输出：den - 大气密度
def ComputeDensity(r):
    p0 = 3.6e-10
    H0 = 37.4
    r0 = 6408.4
    H = H0 + 0.12 * (r - r0)
    den = p0 * np.exp(-(r - r0) / H)
    return den
