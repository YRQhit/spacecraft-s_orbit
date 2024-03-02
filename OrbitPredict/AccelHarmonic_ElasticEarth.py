import numpy as np
import globals
#  输入参数：
#  Mjd - 儒略日(UTC)         r - 地心惯性系下的坐标
#  deg - 非球形摄动阶数      E - 惯性系到地固系的转换矩阵
#  输出参数：
#  a - 非球形摄动产生的三轴加速度
def AccelHarmonic_ElasticEarth(r, deg):

    r_bf = np.matmul(globals.E, r)  # 将卫星位置矢量 r 由地惯系 ICRF 转换为地固系 ITRS
    lon, latgc, d = CalcPolarAngles(r_bf)  # 计算地心纬度、地心经度、地心距

    if lon > np.pi:
        lon = lon - 2 * np.pi

    pnm, dpnm = Legendre(deg, deg, latgc)
    dUdr = 0
    dUdlat = 0
    dUdlon = 0
    q1 = 0
    q2 = 0
    q3 = 0

    for n in range(2, deg + 1):
        b1 = (-globals.GM_Earth / d ** 2) * (globals.r_E / d) ** n * (n + 1)
        b2 = (globals.GM_Earth / d) * (globals.r_E / d) ** n
        b3 = (globals.GM_Earth / d) * (globals.r_E / d) ** n

        for m in range(n + 1):
            ml = m * lon
            q1 = q1 + pnm[n][m] * (globals.S[n][m] * np.sin(ml) + globals.C[n][m] * np.cos(ml))
            q2 = q2 + dpnm[n][m] * (globals.S[n][m] * np.sin(ml) + globals.C[n][m] * np.cos(ml))
            q3 = q3 + m * pnm[n][m] * (globals.S[n][m] * np.cos(ml) - globals.C[n][m] * np.sin(ml))

        dUdr = dUdr + q1 * b1  # U 对地心距 r 的偏导数
        dUdlat = dUdlat + q2 * b2  # U 对地心精度 φ 的偏导数
        dUdlon = dUdlon + q3 * b3  # U 对地心纬度 λ 的偏导数
        q1 = 0
        q2 = 0
        q3 = 0

    x = r_bf[0]
    y = r_bf[1]
    z = r_bf[2]
    xy2 = x ** 2 + y ** 2
    xyn = np.sqrt(xy2)

    R_sph2rec = np.zeros((3, 3))
    R_sph2rec[:, 0] = [x, y, z] / d
    R_sph2rec[:, 1] = [-x * z / xyn, -y * z / xyn, xyn] / d ** 2
    R_sph2rec[:, 2] = [-y, x, 0] / xy2

    a_bf = np.matmul(R_sph2rec, [dUdr, dUdlat, dUdlon])
    a = np.matmul(np.transpose(globals.E), a_bf)
    return a

#  计算地心经纬度
# 输入：r_bf - 地固系下的坐标
#  输出：lon - 地心维度   latgc - 地心经度    d - 地心距

def CalcPolarAngles(r_bf):
    rhoSqr = r_bf[0] * r_bf[0] + r_bf[1] * r_bf[1]
    d = np.sqrt(rhoSqr + r_bf[2] * r_bf[2])

    if r_bf[0] == 0 and r_bf[1] == 0:
        lon = 0
    else:
        lon = np.arctan2(r_bf[1], r_bf[0])

    if lon < 0:
        lon = lon + 2 * np.pi

    rho = np.sqrt(rhoSqr)
    if r_bf[2] == 0 and rho == 0:
        latgc = 0
    else:
        latgc = np.arctan2(r_bf[2], rho)

    return lon, latgc, d

# 计算勒让德参数表
import numpy as np

def Legendre(n, m, fi):
    sf = np.sin(fi)
    cf = np.cos(fi)

    pnm = np.zeros((n+1, m+1))
    dpnm = np.zeros((n+1, m+1))

    pnm[0, 0] = 1
    dpnm[0, 0] = 0
    pnm[1, 1] = np.sqrt(3) * cf
    dpnm[1, 1] = -np.sqrt(3) * sf

    for i in range(2, n+1):
        pnm[i, i] = np.sqrt((2*i+1)/(2*i)) * cf * pnm[i-1, i-1]
        dpnm[i, i] = np.sqrt((2*i+1)/(2*i)) * (cf * dpnm[i-1, i-1] - sf * pnm[i-1, i-1])

    for i in range(1, n+1):
        pnm[i, i-1] = np.sqrt(2*i+1) * sf * pnm[i-1, i-1]
        dpnm[i, i-1] = np.sqrt(2*i+1) * (cf * pnm[i-1, i-1] + sf * dpnm[i-1, i-1])

    j = 0
    k = 2
    while True:
        for i in range(k, n+1):
            pnm[i, j] = np.sqrt((2*i+1)/((i-j)*(i+j))) * ((np.sqrt(2*i-1)*sf*pnm[i-1, j]) - (np.sqrt(((i+j-1)*(i-j-1))/(2*i-3))*pnm[i-2, j]))
            dpnm[i, j] = np.sqrt((2*i+1)/((i-j)*(i+j))) * ((np.sqrt(2*i-1)*sf*dpnm[i-1, j]) + (np.sqrt(2*i-1)*cf*pnm[i-1, j]) - (np.sqrt(((i+j-1)*(i-j-1))/(2*i-3))*dpnm[i-2, j]))

        j += 1
        k += 1
        if j > m:
            break

    return pnm, dpnm
