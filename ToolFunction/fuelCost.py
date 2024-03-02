#  燃料消耗计算
#  输入：dv - 速度增量(km/s)    m - 航天器质量（kg)
#  输出：fuel - 燃料消耗(kg)
import math

def fuelCost(dv, m, I=None):
    if I is None:
        I = 285

    g0 = 9.8
    dv = abs(dv) * 1000
    fuel = m * (1 - math.exp(-dv / I / g0))
    return fuel
