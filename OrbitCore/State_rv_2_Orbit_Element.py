import numpy as np
from OrbitCore.Orbit_Element_2_State_rv import Orbit_Element_2_State_rv
#  函数功能：由航天器位置速度矢量求航天器轨道六要素
#  输入：
#        R：航天器位置矢量（行向量,列向量均可，单位，km）
#        V：航天器速度矢量（行向量,列向量均可，单位km/s）。
#        muu：地球引力常量，缺省输入为398600.4415kg^3/s^2。
#  输出：
#        coe：航天器轨道六要素，具体见下（列向量）。
#  ---------coe（classical orbit elements）------------- %
#  a：轨道半长轴（单位，km）
#  e：轨道偏心率（无量纲）
#  incl：轨道倾角(单位，°)
#  RAAN：升交点赤经(单位，°)
#  omegap：近地点幅角(单位，°)
#  TA：真近点角(单位，°)
def State_rv_2_Orbit_Element(R, V, mu=398600.4415):
    eps = 1e-6
    r2d = 180 / np.pi

    R = np.reshape(R, (3, 1))
    V = np.reshape(V, (3, 1))
    r = np.linalg.norm(R)
    v = np.linalg.norm(V)

    vr = np.dot(np.transpose(R), V) / r  # 径向速度
    H = np.cross(np.transpose(R), np.transpose(V))  # 比角动量
    H = H[0]
    h = np.linalg.norm(H)
    incl = np.arccos(H[2] / h)  # 轨道倾角

    N = np.cross(np.array([0, 0, 1]), H)  # 真近点角
    n = np.linalg.norm(N)

    if np.abs(incl) <= 1e-6:
        RA = 0
    elif n != 0:
        RA = np.arccos(N[0] / n)
        if N[1] < 0:
            RA = 2 * np.pi - RA
    else:
        RA = 0

    E = 1 / mu * ((v ** 2 - mu / r) * R - r * vr * V)
    e = np.linalg.norm(E)

    if np.abs(e) <= 1e-10:
        omegap = 0
    elif n != 0:
        if e > eps:
            omegap = np.real(np.arccos(np.dot(N, E) / (n * e))).item()
            # omegap = np.real(np.arccos(np.dot(N, E) / n / e))
            if E[2] < 0:
                omegap = 2 * np.pi - omegap
        else:
            omegap = 0
    else:
        omegap = 0

    if e > eps:
        TA = np.real(np.arccos(np.dot(E.transpose(), R) / e / r))
        TA = float(TA)
        if vr < 0:
            TA = 2 * np.pi - TA
    else:
        TA = np.real(np.arccos(np.dot(N.transpose(), R) / n / r))
        # TA = np.real(np.arccos(np.dot(N.transpose(), R) / (n + 0.0000000001) / r))
    a = h ** 2 / mu / (1 - e ** 2)
    coe = np.array([a, e, incl * r2d, RA * r2d, omegap * r2d, TA * r2d], dtype=object)
    r, _ = Orbit_Element_2_State_rv(coe)
    r = np.reshape(r,(3,1))
    if np.linalg.norm(r - R) > 1:
        coe[5] = 360 - coe[5]

    if e < eps and incl <= 1e-6:
        omegap = omegap + RA
        RA = 0
        TA = TA + omegap
        omegap = 0
        v_dir = np.cross(np.array([0, 0, 1]), np.array([1, 0, 0]))
        dir_proj = np.sign(np.dot(R.transpose(), v_dir) / np.linalg.norm(v_dir))
        TA = dir_proj * np.arccos(np.dot([1, 0, 0], R) / np.linalg.norm(R))
    # coe[5] = (coe[5])[0]
    return coe
