import numpy as np
from scipy.linalg import expm
from OrbitCore.Orbit_Element_2_State_rv import Orbit_Element_2_State_rv
from ToolFunction.Inertial2Relative import Inertial2Relative
from OrbitPredict.OrbitPrediction import OrbitPrediction
from OrbitalTransfer.CW_Transfer import CWStateTransitionMatrix

def CW2ImpulseTransfer(coe_c, coe_t, delta, t=None):
    GM_Earth = 398600.4418

    n = np.sqrt(GM_Earth / coe_t[0] ** 3)

    if t is None:
        t = np.pi / n

    # Replace Orbit_Element_2_State_rv, Inertial2Relative, and OrbitPrediction with your implementations
    r_c, v_c = Orbit_Element_2_State_rv(coe_c, GM_Earth)
    r_t, v_t = Orbit_Element_2_State_rv(coe_t, GM_Earth)

    p, QXx = Inertial2Relative(r_t, v_t, r_c, v_c)

    [Qrr, Qrv, _, _] = CWStateTransitionMatrix(n, t)


    if np.abs(np.linalg.det(Qrv)) > 1e-3:
        v1 = np.linalg.inv(Qrv).dot(delta[:3] - Qrr.dot(p[:3]))
    else:
        v1 = np.linalg.pinv(Qrv).dot(delta[:3] - Qrr.dot(p[:3]))

    deltv1 = np.dot(QXx.T, v1 - p[3:])

    targetPosVel, _ = OrbitPrediction(np.concatenate((r_t, v_t)), t, 60, [0, 0], 'RK7')
    E = []

    for i in range(10):
        chasePosVel, _ = OrbitPrediction(np.concatenate((r_c, v_c + deltv1)), t, 60, [0, 0], 'RK7')
        relState, QXx2 = Inertial2Relative(targetPosVel[:3], targetPosVel[3:], chasePosVel[:3], chasePosVel[3:])
        err = relState - delta
        E.append(np.linalg.norm(err[:3]))

        if np.linalg.norm(err[:3]) < 0.0001:
            break

        if np.abs(np.linalg.det(Qrv)) > 1e-3:
            v1 = v1 - np.linalg.inv(Qrv).dot(err[:3])
        else:
            v1 = v1 - np.linalg.pinv(Qrv).dot(err[:3])

        deltv1 = np.dot(QXx.T, v1 - p[3:])

    deltv2 = np.dot(QXx2.T, delta[3:] - relState[3:])

    return deltv1, deltv2


# [deltv1, deltv2] = CW2ImpulseTransfer([6885;0.001;97.5;10;20;5], [6885;0.0012;97.5;10;20;5.3], [10;0;0;0;0;0], 5000)

# coe_c = np.array([6885,0.001,97.5,10,20,5])
# coe_t = np.array([6885,0.0012,97.5,10,20,5.3])
# delta = np.array([10,0,0,0,0,0])
# [deltv1, deltv2] = CW2ImpulseTransfer(coe_c, coe_t, delta, 5000)
