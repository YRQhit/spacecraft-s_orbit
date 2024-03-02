# 推力一定计算燃料消耗率 推力T
def kCal(T):
    g0 = 9.8
    Isp = 3000
    k = -T / g0 / Isp
    return k
