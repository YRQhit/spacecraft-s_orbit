#  轨道递推（考虑非球形摄动以及大气摄动）
#  输入：x0 - 初始的位置速度   time - 递推时间
#        h - 递推步长          integrator - 积分器类型（RK4/RK7）
#        model - 是否考虑摄动项（1/0） [非球形 大气]
#        statTime - 初始时刻（给定该值后采用高精度递推）
#  输出：x - 最终的位置速度    xt - 递推产生的中间数据
# 使用方法
#x0 = [42166,0,0,0,3.07459,0]
# startTime = np.array([2019,1,1,0,0,0])
# globals.orbitModel = 'HPOP'
# x,xt = OrbitPrediction(x0,360,60,[1 ,1],'RK7',startTime);
# OrbitPrediction(x0,360,60)
# OrbitPrediction(x0,360,60,[1 ,1 ,1],'RK7',startTime)
import globals
import numpy as np
from OrbitPredict.AccelDrag import AccelDrag
from OrbitPredict.sofa.ICRS2ITRS import ICRS2ITRS
from OrbitPredict.sofa.Mjday import Mjday
from OrbitPredict.AddTime import AddTime
from OrbitPredict.AccelThird import AccelThird
from OrbitPredict.AccelHarmonic_ElasticEarth import AccelHarmonic_ElasticEarth
from OrbitPredict.ComputeDenstiy_HPOP import ComputeDensity_HPOP

# Orbit Prediction function
def OrbitPrediction(x0, time, h, model=[1, 1], integrator='RK4', startTime=None):
    x = np.array(x0)
    xt = np.array(x0)
    xk = np.zeros(4)
    k4 = np.zeros((4, 6))
    k7 = np.zeros((13, 6))
    M_T = 1 / 86400
    Mjd = 0
    h = h * np.sign(time)  # Determine step size based on the direction of propagation
    param_k = np.zeros((13, 12))
    dens = 0
    if (h!=0 and time % h == 0):
        finalStep = h
        num = int(time / h)
    elif (h==0 and time == 0):
        finalStep = np.nan
        num = np.nan
    else:
        finalStep = time % h
        num = int(time / h) + 1





    if startTime is not None:
        year, mon, day, hour, minute, sec = startTime
        Mjd0 = Mjday(year, mon, day, hour, minute, sec)
        Mjd = Mjd0
        E = ICRS2ITRS(Mjd)
        dens = ComputeDensity_HPOP(x0, startTime, 0)

    if integrator == 'RK4':
        param_k = np.array([0, 1 / 2, 1 / 2, 1])
        param_t = np.array([1, 2, 2, 1])
    elif integrator == 'RK7':
        param_t = np.array([0.0, 2.0 / 27.0, 1.0 / 9.0, 1.0 / 6.0, 5.0 / 12.0, 1.0 / 2.0, 5.0 / 6.0,
                            1.0 / 6.0, 2.0 / 3.0, 1.0 / 3.0, 1.0, 0.0, 1.0])
        param_c = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 34.0 / 105.0, 9.0 / 35.0, 9.0 / 35.0,
                            9.0 / 280.0, 9.0 / 280.0, 0.0, 41.0 / 840.0, 41.0 / 840.0])


        param_k[0,:] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        param_k[1,:] = [2.0 / 27.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        param_k[2,:] = [1.0 / 36.0, 1.0 / 12.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        param_k[3,:] = [1.0 / 24.0, 0.0, 1.0 / 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        param_k[4,:] = [5.0 / 12.0, 0.0, -25.0 / 16.0, 25.0 / 16.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        param_k[5,:] = [1.0 / 20.0, 0.0, 0.0, 1.0 / 4.0, 1.0 / 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        param_k[6,:] = [-25.0 / 108.0, 0.0, 0.0, 125.0 / 108.0, -65.0 / 27.0, 125.0 / 54.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0]
        param_k[7,:] = [31.0 / 300.0, 0.0, 0.0, 0.0, 61.0 / 225.0, -2.0 / 9.0, 13.0 / 900.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        param_k[8,:] = [2.0, 0.0, 0.0, -53.0 / 6.0, 704.0 / 45.0, -107.0 / 9.0, 67.0 / 90.0, 3.0, 0.0, 0.0, 0.0, 0.0]
        param_k[9,:] = [-91.0 / 108, 0.0, 0.0, 23.0 / 108.0, -976.0 / 135.0, 311.0 / 54.0, -19.0 / 60.0, 17.0 / 6.0,
                         -1.0 / 12.0, 0.0, 0.0, 0.0]
        param_k[10,:] = [2383.0 / 4100.0, 0.0, 0.0, -341.0 / 164.0, 4496.0 / 1025.0, -301.0 / 82.0, 2133.0 / 4100.0,
                         45.0 / 82.0, 45.0 / 164.0, 18.0 / 41.0, 0.0, 0.0]
        param_k[11,:] = [3.0 / 205.0, 0.0, 0.0, 0.0, 0.0, -6.0 / 41.0, -3.0 / 205.0, -3.0 / 41.0, 3.0 / 41.0,
                         6.0 / 41.0, 0.0, 0.0]
        param_k[12,:] = [-1777.0 / 4100.0, 0.0, 0.0, -341.0 / 164.0, 4496.0 / 1025.0, -289.0 / 82.0, 2193.0 / 4100.0,
                         51.0 / 82.0, 33.0 / 164.0, 12.0 / 41.0, 0.0, 1.0]
    if integrator == 'RK4':
        if(np.isnan(num)==True):
            for i in range(1):
                if i == num - 1:
                    h = finalStep

                for j in range(4):
                    xk = np.zeros(6)
                    for k in range(6):
                        if j > 0:
                            xk[k] = param_k[j] * h * k4[j - 1, k]

                    k4[j, 0] = x[3] + xk[3]
                    k4[j, 1] = x[4] + xk[4]
                    k4[j, 2] = x[5] + xk[5]

                    a = np.array([0, 0, 0])
                    if startTime is None:  # Low-accuracy propagation
                        a = Accel(x, xk, model)
                    elif startTime is not None and globals.orbitModel == "LPOP":
                        a = Accel(x, xk, model, dens)

                    elif startTime is not None and globals.orbitModel == "HPOP":  # High-accuracy propagation
                        globals.E = ICRS2ITRS(Mjd + M_T * h * param_k[j])
                        # 二体引力
                        r = np.sqrt((x[0] + xk[0]) ** 2 + (x[1] + xk[1]) ** 2 + (x[2] + xk[2]) ** 2) ** 1.5
                        a = -globals.GM_Earth / r * np.array([x[0] + xk[0], x[1] + xk[1], x[2] + xk[2]])
                        # 非球型摄动
                        if model[0] == 1:  # Non-spherical perturbations
                            R = np.array([x[0] + xk[0], x[1] + xk[1], x[2] + xk[2]])
                            a += AccelHarmonic_ElasticEarth(R, globals.deg)
                        # 大气摄动
                        if model[1] == 1:  # Atmospheric drag
                            dens = ComputeDensity_HPOP(x + xk, startTime, Mjd - Mjd0 + M_T * h * param_k[j])
                            a += AccelDrag(x + xk, dens)
                        # 三体摄动
                        if len(model) == 3 and model[2] == 1:  # Third-body perturbations
                            a += AccelThird(x[:3] + xk[:3], Mjd + M_T * h * param_k[j])

                    k4[j, 3] = a[0]
                    k4[j, 4] = a[1]
                    k4[j, 5] = a[2]

                Mjd += M_T * h
                x += h * np.dot(k4.T, param_t) / 6
                xt = np.hstack((xt, x))
        else:
            for i in range(num):
                if i == num - 1:
                    h = finalStep

                for j in range(4):
                    xk = np.zeros(6)
                    for k in range(6):
                        if j > 0:
                            xk[k] = param_k[j] * h * k4[j - 1, k]

                    k4[j, 0] = x[3] + xk[3]
                    k4[j, 1] = x[4] + xk[4]
                    k4[j, 2] = x[5] + xk[5]

                    a = np.array([0, 0, 0])
                    if startTime is None:  # Low-accuracy propagation
                        a = Accel(x, xk, model)
                    elif startTime is not None and globals.orbitModel == "LPOP":
                        a = Accel(x, xk, model, dens)

                    elif startTime is not None and globals.orbitModel == "HPOP":  # High-accuracy propagation
                        globals.E = ICRS2ITRS(Mjd + M_T * h * param_k[j])
                        # 二体引力
                        r = np.sqrt((x[0] + xk[0]) ** 2 + (x[1] + xk[1]) ** 2 + (x[2] + xk[2]) ** 2) ** 1.5
                        a = -globals.GM_Earth / r * np.array([x[0] + xk[0], x[1] + xk[1], x[2] + xk[2]])
                        # 非球型摄动
                        if model[0] == 1:  # Non-spherical perturbations
                            R = np.array([x[0] + xk[0], x[1] + xk[1], x[2] + xk[2]])
                            a += AccelHarmonic_ElasticEarth(R, globals.deg)
                        # 大气摄动
                        if model[1] == 1:  # Atmospheric drag
                            dens = ComputeDensity_HPOP(x + xk, startTime, Mjd - Mjd0 + M_T * h * param_k[j])
                            a += AccelDrag(x + xk, dens)
                        # 三体摄动
                        if len(model) == 3 and model[2] == 1:  # Third-body perturbations
                            a += AccelThird(x[:3] + xk[:3], Mjd + M_T * h * param_k[j])

                    k4[j, 3] = a[0]
                    k4[j, 4] = a[1]
                    k4[j, 5] = a[2]

                Mjd += M_T * h
                x += h * np.dot(k4.T, param_t) / 6
                xt = np.hstack((xt, x))

    elif integrator == "RK7":
        for i in range(num):
            if i == num - 1:
                h = finalStep

            for j in range(13):
                xk = np.array([0,0,0,0,0,0])
                xk = xk.astype('float64')
                if j > 0:
                    for n in range(j):
                        xk += param_k[j,n]  * k7[n,:]
                k7[j, 0] = h * (x[3] + xk[3])
                k7[j, 1] = h * (x[4] + xk[4])
                k7[j, 2] = h * (x[5] + xk[5])
                a = np.array([0, 0, 0])
                if startTime is None:  # Low-accuracy propagation
                    a = h * Accel(x, xk, model)
                elif startTime is not None and globals.orbitModel=="LPOP":
                    a = h * Accel(x, xk, model, dens)

                elif startTime is not None and globals.orbitModel=="HPOP":  # High-accuracy propagation
                    globals.E = ICRS2ITRS(Mjd + M_T * h * param_t[j])
                    # 二体引力
                    #这个轨道部分有点问题
                    r = ((x[0] + xk[0]) ** 2 + (x[1] + xk[1]) ** 2 + (x[2] + xk[2]) ** 2)**1.5
                    # r = np.sqrt((x[0] + xk[0]) ** 2 + (x[1] + xk[1]) ** 2 + (x[2] + xk[2]) ** 2)
                    a = -h * globals.GM_Earth / r * np.array([x[0] + xk[0], x[1] + xk[1], x[2] + xk[2]])
                    # 非球型摄动
                    if model[0] == 1:  # Non-spherical perturbations
                        R = np.array([x[0] + xk[0], x[1] + xk[1], x[2] + xk[2]])
                        a += h * AccelHarmonic_ElasticEarth(R, globals.deg)
                    # 大气摄动
                    if model[1] == 1:  # Atmospheric drag
                        dens = ComputeDensity_HPOP(x + xk, startTime, Mjd - Mjd0 )
                        a += h * AccelDrag(x + xk, dens)
                    # 三体摄动
                    if len(model) == 3 and model[2] == 1:  # Third-body perturbations
                        a += h * AccelThird(x[:3] + xk[:3], Mjd + M_T * h * param_t[j])

                k7[j, 3] = a[0]
                k7[j, 4] = a[1]
                k7[j, 5] = a[2]

            Mjd += M_T * h
            x += k7.transpose() @ param_c
            # xt = np.tile(xt, x)
            xt = np.vstack((xt, x))
    return x,xt


def Accel(x, xk, model, dens=None):

    r = np.sqrt((x[0] + xk[0]) ** 2 + (x[1] + xk[1]) ** 2 + (x[2] + xk[2]) ** 2)
    a = np.zeros(3)

    if model[0] == 0:
        a = -globals.GM_Earth / r ** 3 * np.array([x[0] + xk[0], x[1] + xk[1], x[2] + xk[2]])
    elif model[0] == 1:
        a[0] = -globals.GM_Earth * (x[0] + xk[0]) / r ** 3 * (
                    1 + 1.5 * globals.J2 * (globals.r_E / r) ** 2 * (1 - 5 * (x[2] + xk[2]) ** 2 / r ** 2))
        a[1] = a[0] * (x[1] + xk[1]) / (x[0] + xk[0])
        a[2] = -globals.GM_Earth * (x[2] + xk[2]) / r ** 3 * (
                    1 + 1.5 * globals.J2 * (globals.r_E / r) ** 2 * (3 - 5 * (x[2] + xk[2]) ** 2 / r ** 2))

    if model[1] == 1:
        if dens is None:
            a = a + AccelDrag(x + xk)
        else:
            a = a + AccelDrag(x + xk, dens)

    return a

