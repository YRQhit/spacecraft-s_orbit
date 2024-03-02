import numpy as np


def CW_Transfer(r_LVLH, v_LVLH, n, t):
    Qrr, Qrv, Qvr, Qvv = CWStateTransitionMatrix(n, t)
    rt_LVLH = np.dot(Qrr, r_LVLH) + np.dot(Qrv, v_LVLH)
    vt_LVLH = np.dot(Qvr, r_LVLH) + np.dot(Qvv, v_LVLH)
    return rt_LVLH, vt_LVLH

#输入：n - 轨道角速度             t - 时间
def CWStateTransitionMatrix(n, t):
    Qrr = np.array([[4 - 3 * np.cos(n * t), 0, 0],
                    [6 * (np.sin(n * t) - n * t), 1, 0],
                    [0, 0, np.cos(n * t)]])

    Qrv = np.array([[np.sin(n * t) / n, 2 * (1 - np.cos(n * t)) / n, 0],
                    [2 * (np.cos(n * t) - 1) / n, (4 * np.sin(n * t) - 3 * n * t) / n, 0],
                    [0, 0, np.sin(n * t) / n]])

    Qvr = np.array([[3 * n * np.sin(n * t), 0, 0],
                    [6 * n * (np.cos(n * t) - 1), 0, 0],
                    [0, 0, -n * np.sin(n * t)]])

    Qvv = np.array([[np.cos(n * t), 2 * np.sin(n * t), 0],
                    [-2 * np.sin(n * t), 4 * np.cos(n * t) - 3, 0],
                    [0, 0, np.cos(n * t)]])

    return Qrr, Qrv, Qvr, Qvv


# # 示例用法
# r_LVLH = np.array([1, 2, 3])
# v_LVLH = np.array([0.1, 0.2, 0.3])
# n = 0.1
# t = 2.0
#
# rt_LVLH, vt_LVLH = CW_Transfer(r_LVLH, v_LVLH, n, t)
# print("rt_LVLH:", rt_LVLH)
# print("vt_LVLH:", vt_LVLH)
