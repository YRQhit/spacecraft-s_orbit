#  经纬高转惯性系
#  输入：Mjd - 儒略时间  lonlan - 经度/纬度/轨高
#  输出：pos - 惯性系下位置
import numpy as np
from OrbitPredict.sofa.ICRS2ITRS import ICRS2ITRS
import globals
def LLA2ICRS(Mjd, lonlan):

    f = 1 / 298.257223563  # 地球扁率，无量纲
    R_p = (1 - f) * globals.r_E  # 地球极半径
    e1 = np.sqrt(globals.r_E * globals.r_E - R_p * R_p) / globals.r_E
    lon = lonlan[0]
    lan = lonlan[1]
    alt = lonlan[2]
    temp = globals.r_E / np.sqrt(1 - e1 * e1 * np.sin(np.radians(lan)) * np.sin(np.radians(lan)))
    pos = np.zeros(3)
    pos[0] = (temp + alt) * np.cos(np.radians(lan)) * np.cos(np.radians(lon))
    pos[1] = (temp + alt) * np.cos(np.radians(lan)) * np.sin(np.radians(lon))
    pos[2] = (temp * (1 - e1 * e1) + alt) * np.sin(np.radians(lan))

    E = ICRS2ITRS(Mjd)  # You need to implement or import ICRS2ITRS function
    pos = np.dot(E.T, pos)

    return pos

# # Example values
# Mjd = 58650.0  # Example Modified Julian Date (MJD)
# lonlan = [120.0, 30.0, 1000.0]  # Example lon, lat, alt in degrees and meters
#
# # Call the LLA2ICRS function
# pos_icrs = LLA2ICRS(Mjd, lonlan)
#
# # Print the resulting ICRS position
# print("ICRS Position:", pos_icrs)
