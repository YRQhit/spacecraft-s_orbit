import numpy as np


def Inertial2Orbit(RV):
    R = RV[0:3]
    V = RV[3:6]

    if np.linalg.norm(R) <= 0:
        raise ValueError('Satellite Position norm(R) = 0 in Inertial2Orbit!')

    k = -R / np.linalg.norm(R)

    H = np.cross(R, V)
    j = -H / np.linalg.norm(H)
    j = j / np.linalg.norm(j)
    i = np.cross(j, k)
    i = i / np.linalg.norm(i)

    L_oi = np.vstack((i, j, k))
    return L_oi
