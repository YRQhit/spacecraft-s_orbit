import math
import globals
# % 真近点角转平近点角
# % 输入：e - 偏心率  trueAno - 真近点角（deg)
# % 输出：Me - 平近点角（deg)
def true2Mean(e, trueAno):
    trueAno = trueAno * globals.deg2rad
    E = 2 * math.atan(math.sqrt((1 - e) / (1 + e)) * math.tan(trueAno / 2))
    Me = (E - e * math.sin(E)) * globals.rad2deg
    return Me
