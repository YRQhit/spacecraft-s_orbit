# --------------------------------------------------------------------------
#
#  JPL_Eph_DE405: Computes the sun, moon, and nine major planets' equatorial
#                 position using JPL Ephemerides
#
#  Inputs:
#    Mjd_UTC         Modified julian date of UTC
#
#  Output:return r_Earth, r_Moon, r_Sun,  r_Mars和matlab的程序不一样，此处只有地球，月球，太阳，火星的r
########下面的是matlab的output
#    r_Mercury,r_Venus,r_Earth(solar system barycenter (SSB)),r_Mars,
#    r_Jupiter,r_Saturn,r_Uranus,r_Neptune,r_Pluto,r_Moon,
#    r_Sun(geocentric equatorial position ([m]) referred to the
#    International Celestial Reference Frame (ICRF))
#
#  Notes: Light-time is already taken into account
#
#  Last modified:   2015/08/12   M. Mahooti
#
# --------------------------------------------------------------------------
import numpy as np
import globals

def Cheb3D(t, N, Ta, Tb, Cx, Cy, Cz):
    # Check validity
    if t < Ta or t > Tb:
        raise ValueError('ERROR: Time out of range in cheb3d')
    # Clenshaw algorithm
    tau = (2 * t - Ta - Tb) / (Tb - Ta)

    f1 = np.zeros(3)
    f2 = np.zeros(3)

    for i in range(N, 1, -1):
        old_f1 = f1
        f1 = 2 * tau * f1 - f2 + [Cx[i - 1], Cy[i - 1], Cz[i - 1]]  # 注意Python中列表索引是从0开始的，所以索引需要减1
        f2 = old_f1

    ChebApp = tau * f1 - f2 + np.array([Cx[0], Cy[0], Cz[0]])

    return ChebApp



def JPL_Eph_DE405(Mjd_UTC):


    JD = Mjd_UTC + 2400000.5

    for i in range(1147):
        if (globals.PC[i][ 0] <= JD and JD <= globals.PC[i][ 1]):
            PCtemp = globals.PC[i][ :]

    t1 = PCtemp[0] - 2400000.5  # MJD at start of interval

    dt = Mjd_UTC - t1

    temp = np.arange(231, 271, 13)
    Cx_Earth = PCtemp[temp[0] - 1 : temp[1] - 1]
    Cy_Earth = PCtemp[temp[1] - 1:temp[2] - 1]
    Cz_Earth = PCtemp[temp[2] - 1:temp[3] - 1]
    temp = temp + 39
    Cx = PCtemp[temp[0] - 1:temp[1] - 1]
    Cy = PCtemp[temp[1] - 1:temp[2] - 1]
    Cz = PCtemp[temp[2] - 1:temp[3] - 1]
    Cx_Earth = np.concatenate([Cx_Earth, Cx])
    Cy_Earth = np.concatenate([Cy_Earth, Cy])
    Cz_Earth = np.concatenate([Cz_Earth, Cz])
    if (0 <= dt and dt <= 16):
        j = 0
        Mjd0 = t1
    elif (16 < dt and dt <= 32):
        j = 1
        Mjd0 = t1 + 16 * j
    r_Earth = 1e3 * Cheb3D(Mjd_UTC, 13, Mjd0, Mjd0 + 16, Cx_Earth[13 * j :13 * j + 13],
                           Cy_Earth[13 * j :13 * j + 13], Cz_Earth[13 * j :13 * j + 13])
    temp = np.arange(441, 481, 13)
    Cx_Moon = PCtemp[temp[0] - 1:temp[1] - 1]
    Cy_Moon = PCtemp[temp[1] - 1:temp[2] - 1]
    Cz_Moon = PCtemp[temp[2] - 1:temp[3] - 1]
    for i in range(7):
        temp = temp + 39
        Cx = PCtemp[temp[0] - 1:temp[1] - 1]
        Cy = PCtemp[temp[1] - 1:temp[2] - 1]
        Cz = PCtemp[temp[2] - 1:temp[3] - 1]

        Cx_Moon = np.concatenate([Cx_Moon, Cx])
        Cy_Moon = np.concatenate([Cy_Moon, Cy])
        Cz_Moon = np.concatenate([Cz_Moon, Cz])
    if (0 <= dt and dt <= 4):
        j = 0
        Mjd0 = t1
    elif (4 < dt and dt <= 8):
        j = 1
        Mjd0 = t1 + 4 * j
    elif (8 < dt and dt <= 12):
        j = 2
        Mjd0 = t1 + 4 * j
    elif (12 < dt and dt <= 16):
        j = 3
        Mjd0 = t1 + 4 * j
    elif(16 < dt and dt <= 20):
        j = 4
        Mjd0 = t1 + 4 * j
    elif(20 < dt and dt <= 24):
        j = 5
        Mjd0 = t1 + 4 * j
    elif(24 < dt and dt <= 28):
        j = 6
        Mjd0 = t1 + 4 * j
    elif(28 < dt and dt <= 32):
        j = 7
        Mjd0 = t1 + 4 * j
    r_Moon = 1e3 * Cheb3D(Mjd_UTC, 13, Mjd0, Mjd0 + 4, Cx_Moon[13 * j :13 * j + 13],
                          Cy_Moon[13 * j :13 * j + 13], Cz_Moon[13 * j :13 * j + 13])

    temp = np.arange(753, 787, 11)
    Cx_Sun = PCtemp[temp[0] - 1:temp[1] - 1]
    Cy_Sun = PCtemp[temp[1] - 1:temp[2] - 1]
    Cz_Sun = PCtemp[temp[2] - 1:temp[3] - 1]
    temp = temp + 33
    Cx = PCtemp[temp[0] - 1:temp[1] - 1]
    Cy = PCtemp[temp[1] - 1:temp[2] - 1]
    Cz = PCtemp[temp[2] - 1:temp[3] - 1]
    Cx_Sun = np.concatenate([Cx_Sun , Cx])
    Cy_Sun = np.concatenate([Cy_Sun , Cy])
    Cz_Sun = np.concatenate([Cz_Sun , Cz])

    if (0 <= dt and dt <= 16):
        j = 0
        Mjd0 = t1
    elif (16 < dt and dt <= 32):
        j = 1
        Mjd0 = t1 + 16 * j

    r_Sun = 1e3 * Cheb3D(Mjd_UTC, 11, Mjd0, Mjd0 + 16, Cx_Sun[11 * j :11 * j + 11],
                         Cy_Sun[11 * j :11 * j + 11], Cz_Sun[11 * j :11 * j + 11])

    temp = np.arange(3, 46, 14)
    Cx_Mercury = PCtemp[temp[0] - 1 :temp[1] - 1]
    Cy_Mercury = PCtemp[temp[1] - 1 :temp[2] - 1]
    Cz_Mercury = PCtemp[temp[2] - 1 :temp[3] - 1]
    for i in range(3):
        temp = temp + 42
        Cx = PCtemp[temp[0] - 1 :temp[1] - 1]
        Cy = PCtemp[temp[1] - 1 :temp[2] - 1]
        Cz = PCtemp[temp[2] - 1 :temp[3] - 1]
        Cx_Mercury = np.concatenate([Cx_Mercury, Cx])
        Cy_Mercury = np.concatenate([Cy_Mercury, Cy])
        Cz_Mercury = np.concatenate([Cz_Mercury, Cz])
    if (0 <= dt and dt <= 8):
        j = 0
        Mjd0 = t1
    elif (8 < dt and dt <= 16):
        j = 1
        Mjd0 = t1 + 8 * j
    elif (16 < dt and dt <= 24):
        j = 2
        Mjd0 = t1 + 8 * j
    elif (24 < dt and dt <= 32):
        j = 3
        Mjd0 = t1 + 8 * j

    r_Mercury = 1e3 * Cheb3D(Mjd_UTC, 14, Mjd0, Mjd0 + 8, Cx_Mercury[14 * j :14 * j + 14],
                             Cy_Mercury[14 * j :14 * j + 14], Cz_Mercury[14 * j :14 * j + 14])

    temp = np.arange(171, 202, 10)
    Cx_Venus = PCtemp[temp[0] - 1:temp[1] - 1]
    Cy_Venus = PCtemp[temp[1] - 1:temp[2] - 1]
    Cz_Venus = PCtemp[temp[2] - 1:temp[3] - 1]

    temp = temp + 30
    Cx = PCtemp[temp[0] - 1:temp[1] - 1]
    Cy = PCtemp[temp[1] - 1:temp[2] - 1]
    Cz = PCtemp[temp[2] - 1:temp[3] - 1]
    Cx_Venus = np.concatenate([Cx_Venus, Cx])
    Cy_Venus = np.concatenate([Cy_Venus, Cy])
    Cz_Venus = np.concatenate([Cz_Venus, Cz])
    if (0 <= dt and dt <= 16):
        j = 0
        Mjd0 = t1
    elif (16 < dt and dt <= 32):
        j = 1
        Mjd0 = t1 + 16 * j

    r_Venus = 1e3 * Cheb3D(Mjd_UTC, 10, Mjd0, Mjd0 + 16, Cx_Venus[10 * j :10 * j + 10],
                           Cy_Venus[10 * j :10 * j + 10], Cz_Venus[10 * j :10 * j + 10])

    temp = np.arange(309, 343, 11)
    Cx_Mars = PCtemp[temp[0] - 1:temp[1] - 1]
    Cy_Mars = PCtemp[temp[1] - 1:temp[2] - 1]
    Cz_Mars = PCtemp[temp[2] - 1:temp[3] - 1]
    j = 0
    Mjd0 = t1
    r_Mars = 1e3 * Cheb3D(Mjd_UTC, 11, Mjd0, Mjd0 + 32, Cx_Mars[11 * j :11 * j + 11],
                          Cy_Mars[11 * j :11 * j + 11], Cz_Mars[11 * j :11 * j + 11])

    temp = np.arange(342, 367, 8)
    Cx_Jupiter = PCtemp[temp[0] - 1:temp[1] - 1]
    Cy_Jupiter = PCtemp[temp[1] - 1:temp[2] - 1]
    Cz_Jupiter = PCtemp[temp[2] - 1:temp[3] - 1]
    j = 0
    Mjd0 = t1

    r_Jupiter = 1e3 * Cheb3D(Mjd_UTC, 8, Mjd0, Mjd0 + 32, Cx_Jupiter[8 * j :8 * j + 8],
                             Cy_Jupiter[8 * j :8 * j + 8], Cz_Jupiter[8 * j :8 * j + 8])

    EMRAT = 81.3005600000000044
    EMRAT1 = 1 / (1 + EMRAT)
    r_Earth = r_Earth - EMRAT1 * r_Moon
    r_Sun = -r_Earth + r_Sun
    r_Mars = -r_Earth + r_Mars

    return r_Earth, r_Moon, r_Sun,  r_Mars

