import globals
import numpy as np
from ToolFunction.Inertial2Orbit import Inertial2Orbit
#  考虑J2摄动
#  Thrust_f 为推力大小 推力大小恒定

def J2Cal(t, RV, Thrust_f, deg):
    GM_Earth, J2, r_E = globals.GM_Earth, globals.J2, globals.r_E
    Azimuth = deg[0]
    Elevation = deg[1]

    x = RV[0]
    y = RV[1]
    z = RV[2]

    RotMat = np.transpose(Inertial2Orbit(RV))
    vec = np.array([np.cos(np.radians(Elevation)) * np.cos(np.radians(Azimuth)),
                    np.cos(np.radians(Elevation)) * np.sin(np.radians(Azimuth)),
                    np.sin(np.radians(Elevation))])

    r = np.linalg.norm(RV[0:3])
    dx = -GM_Earth * x / r ** 3 * (1 + 1.5 * J2 * (r_E / r) ** 2 * (1 - 5 * z ** 2 / r ** 2)) + Thrust_f * np.dot(RotMat[0], vec)
    dy = -GM_Earth * x / r ** 3 * (1 + 1.5 * J2 * (r_E / r) ** 2 * (1 - 5 * z ** 2 / r ** 2)) * (y / x) + Thrust_f * np.dot(RotMat[1], vec)
    dz = -GM_Earth * z / r ** 3 * (1 + 1.5 * J2 * (r_E / r) ** 2 * (3 - 5 * z ** 2 / r ** 2)) + Thrust_f * np.dot(RotMat[2], vec)

    drv = [RV[3], RV[4], RV[5], dx, dy, dz]
    return drv
