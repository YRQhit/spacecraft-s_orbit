import math
from OrbitCore.amendDeg import amendDeg
# % 平近点角转真近点角
# % 输入：e - 偏心率       Me - 平近点角（deg)
# % 输出：trueAng - 真近点角（deg)
import globals

def mean2True(e, Me):
    Me_r = amendDeg(Me, '0 - 360') * globals.deg2rad
    E_r = Me_r
    if Me_r < math.pi:
        E_r += e / 2
    else:
        E_r -= e / 2

    ratio = 1
    j = 0
    eps = 1e-10
    while abs(ratio) > eps:
        ratio = (E_r - e * math.sin(E_r) - Me_r) / (1 - e * math.cos(E_r))
        E_r -= ratio
        j += 1
        if j >= 50:
            break

    trueAng = 2 * math.atan(math.sqrt((1 + e) / (1 - e)) * math.tan(E_r / 2)) * globals.rad2deg
    trueAng = amendDeg(trueAng, '0 - 360')

    return trueAng