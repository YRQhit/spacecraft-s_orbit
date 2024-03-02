#航天器2相对于航天器1的轨道根数在1的轨道系下的表示
import numpy as np


def Inertial2Relative(R1, V1, R2, V2):
    mu = 3.986005e5
    rad_to_deg = 180 / np.pi
    deg_to_rad = np.pi / 180

    a1 = -mu * R1 / (np.linalg.norm(R1)) ** 3
    a2 = -mu * R2 / (np.linalg.norm(R2)) ** 3

    i = R1 / np.linalg.norm(R1)
    h1 = np.cross(R1, V1)
    k = h1 / np.linalg.norm(h1)
    j = np.cross(k, i)

    OMG1 = h1 / (np.linalg.norm(R1)) ** 2

    deltaR = R2 - R1
    deltaV = (V2 - V1) - np.cross(OMG1, deltaR)

    QXx = np.array([i, j, k])
    deltaR = np.dot(QXx, deltaR)
    deltaV = np.dot(QXx, deltaV)

    delta = np.hstack((deltaR, deltaV))

    return delta, QXx


# # 测试用例
# R1 = np.array([8000, 10000, 15000])  # km
# V1 = np.array([7, 7.5, 6])           # km/s
# R2 = np.array([8500, 9500, 14500])  # km
# V2 = np.array([7.2, 7.7, 6.2])       # km/s
#
# delta, QXx = Inertial2Relative(R1, V1, R2, V2)
#
# print("Relative Delta:")
# print(delta)
# print("Transformation Matrix QXx:")
# print(QXx)