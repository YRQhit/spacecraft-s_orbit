import numpy as np
import globals
from OrbitCore.true2Mean import true2Mean
from OrbitCore.mean2True import mean2True
from OrbitCore.amendDeg import amendDeg
#考虑J2摄动的轨道地推，输入是六根数，输出六根
def J2Orbit(coe0, T):
    global GM_Earth, J2, r_E, rad2deg
    GM_Earth, J2, r_E, rad2deg = globals.GM_Earth, globals.J2, globals.r_E, globals.rad2deg
    a = coe0[0]
    e = coe0[1]
    i = coe0[2]

    q = true2Mean(coe0[1], coe0[5])

    p = a * (1 - e ** 2)
    n = np.sqrt(GM_Earth / a ** 3)
    k1 = -1.5 * J2 * (r_E / p) ** 2 * n * np.cos(np.deg2rad(i))
    k2 = 1.5 * J2 * (r_E / p) ** 2 * n * (2 - 2.5 * np.sin(np.deg2rad(i)) ** 2)
    k3 = n + 1.5 * J2 * (r_E / p) ** 2 * n * (1 - 1.5 * np.sin(np.deg2rad(i)) ** 2) * np.sqrt(1 - e ** 2)

    coe = coe0.copy()
    coe[3] = amendDeg(coe0[3] + k1 * T * rad2deg, '0 - 360')
    coe[4] = amendDeg(coe0[4] + k2 * T * rad2deg, '0 - 360')
    coe[5] = amendDeg(q + k3 * T * rad2deg, '0 - 360')
    coe[5] = mean2True(e, coe[5])

    return coe
