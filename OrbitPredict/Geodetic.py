import math
import globals
import numpy as np

 # Geodetic: geodetic coordinates (Longitude [rad], latitude [rad],
 #           altitude [m]) from given position vector (r [m])

def Geodetic(r):

    f = 1 / 298.257223563

    epsRequ = np.finfo(float).eps * globals.r_E * 1000  # 收敛准则
    e2 = f * (2 - f)  # Square of eccentricity

    X = r[0] * 1000  # Cartesian coordinates
    Y = r[1] * 1000
    Z = r[2] * 1000
    rho2 = X * X + Y * Y  # Square of distance from z-axis

    # Check validity of input data
    if (np.linalg.norm(r)== 0):
        print('Invalid input in Geodetic constructor')
        lon = 0
        lat = 0
        h = -globals.r_E
        return lon, lat, h

    # Iteration
    dZ = e2 * Z

    while True:
        ZdZ = Z + dZ
        Nh = math.sqrt(rho2 + ZdZ * ZdZ)
        SinPhi = ZdZ / Nh  # Sine of geodetic latitude
        N = globals.r_E * 1000 / math.sqrt(1 - e2 * SinPhi * SinPhi)
        dZ_new = N * e2 * SinPhi
        if math.isclose(abs(dZ - dZ_new), 0, abs_tol=epsRequ):
            break
        dZ = dZ_new

    # Longitude, latitude, altitude
    lon = math.atan2(Y, X) * 180 / math.pi
    lat = math.atan2(ZdZ, math.sqrt(rho2)) * 180 / math.pi
    h = Nh - N

    return lon, lat, h
