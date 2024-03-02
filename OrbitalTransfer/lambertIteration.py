import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
from OrbitPredict.OrbitPrediction import OrbitPrediction
from OrbitalTransfer.lambertOptimal import lamberthigh
import globals

def lambertIteration(r1, r2, T, startTime=None):
    GM_Earth  = globals.GM_Earth
    origin = r1[0:3]
    target = r2
    E = []
    errPre = np.linalg.norm(np.array(origin) - np.array(target))

    if len(r1) == 6:
        v_ref = r1[3:6]
    else:
        v_ref = np.array([0, 0, 0])
    r1_v1 = np.cross(origin, v_ref) / np.linalg.norm(np.cross(origin, v_ref))

    for i in range(10):
        v1, v2 = lamberthigh(origin, target, T)

        norm_diff_1 = np.linalg.norm(np.array(v1) - np.array(v_ref))
        norm_diff_2 = np.linalg.norm(np.array(v2) - np.array(v_ref))

        if norm_diff_1 <= norm_diff_2:
            V1 = v1
        else:
            V1 = v2

        initState = np.hstack((origin, V1))
        if startTime is None:
            finalState, _ = OrbitPrediction(initState, T, 60, [1, 1], 'RK7')
        else:
            finalState, _ = OrbitPrediction(initState, T, 60, [1, 1, 1], 'RK7', startTime)

        err = finalState[0:3] - r2
        e_n = np.linalg.norm(err)

        if errPre > e_n:
            V1Min = V1
            V2Min = finalState[3:6]
            errPre = e_n

        target = target - err
        E.append(e_n)
        if e_n < 0.01:
            break

    return V1Min, V2Min, E


def DeadZone(r1_v1, r2):
    angle = np.dot(r2, r1_v1)
    r = r2 - angle * r1_v1
    return r

# start_time = [2022, 9, 9, 0, 0, 0]
# r2 = [-2.719328240311940e+04, 3.257002397590526e+04, 0.568397574969637, -2.360342780120658, -1.940086718474888, -3.386452149363278e-05]
# r1 = [3.925028471108628e+04, 1.428593531913334e+04, 0.249336608071141, -1.051625445598732, 2.920064636113321, 5.096474227640248e-05]
# T = 25000
# V1Min, V2Min, E = lambertIteration(r1, r2[0:3], T, start_time)
# print(V1Min, V2Min, E)
