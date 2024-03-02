#计算太阳光照角（太阳指向目标星的矢量在目标星轨道平面的投影与追踪星到目标星的矢量夹角）
# 输入：x_c - 追踪星位置速度                    x_t - 目标星位置速度
#      JD_startTime - 儒略日或时间行向量        second - 偏移时间（可以不给定）
# 输出：idealDegree - 理想太阳光照角（默认两星对地心夹角为0）     actualDegree - 实际太阳光照角
import numpy as np
from OrbitCore.CalAstroBodyPos import CalAstroBodyPos
from OrbitPredict.sofa.Mjday import Mjday

def cross_product(a, b):
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])

def IlluminationAngle(x_c, x_t, JD_startTime, second=None):
    if len(JD_startTime) == 6:
        JD = Mjday(JD_startTime[0], JD_startTime[1], JD_startTime[2], JD_startTime[3], JD_startTime[4], JD_startTime[5]) + 2400000.5
    else:
        JD = JD_startTime

    if second is not None:
        JD = JD + second / 86400

    t_norm_vector = cross_product(x_t[:3], x_t[3:]) / np.linalg.norm(cross_product(x_t[:3], x_t[3:]))
    sun_earth_vector = -CalAstroBodyPos(JD)  # 太阳指向地球矢量
    sun_target_vector = x_t[:3] + sun_earth_vector  # 太阳指向目标星矢量

    idealDegree = np.degrees(np.arccos(np.dot(x_t[:3], sun_target_vector) / (np.linalg.norm(x_t[:3]) * np.linalg.norm(sun_target_vector))))

    chase_target_vector = x_t[:3] - x_c[:3]  # 追踪星指向目标星矢量
    actualDegree = np.degrees(np.arccos(np.dot(chase_target_vector, sun_target_vector) / (np.linalg.norm(chase_target_vector) * np.linalg.norm(sun_target_vector))))

    return idealDegree, actualDegree

# 示例用法
# JD_startTime = [2023, 8, 2, 12, 30, 0]  # 日期对应的年月日时分秒
# x_c = np.array([0, 0, 0, 0, 0, 0])  # 追踪星的状态向量
# x_t = np.array([1, 0, 0, 0, 1, 0])  # 目标星的状态向量
# idealDegree, actualDegree = IlluminationAngle(x_c, x_t, JD_startTime)
# print("Ideal Illumination Angle:", idealDegree)
# print("Actual Illumination Angle:", actualDegree)
