import numpy as np

#输入脉冲矢量
#输出 Azimuth-俯仰角，Elevation-方位角
def ThrustAngleCal2(vec):
    z = np.array([0, 0, 1])
    x = np.array([1, 0, 0])

    Elevation = np.degrees(np.arccos(np.dot(vec, z) / np.linalg.norm(vec)))

    if Elevation < 90:
        Elevation = 90 - Elevation
    else:
        Elevation = -Elevation + 90

    vec_z = np.array([0, 0, np.dot(vec, z)])
    vec_xy = vec - vec_z

    Azimuth = np.degrees(np.arccos(np.dot(x, vec_xy) / np.linalg.norm(vec_xy)))

    return Azimuth, Elevation
