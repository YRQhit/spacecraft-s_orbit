import numpy as np
from OrbitPredict.J2Orbit import J2Orbit
from OrbitCore.State_rv_2_Orbit_Element import State_rv_2_Orbit_Element
from OrbitCore.Orbit_Element_2_State_rv import Orbit_Element_2_State_rv
import globals
def J2OrbitRV(rv0, T):
    GM_Earth = globals.GM_Earth
    coe0 = State_rv_2_Orbit_Element(rv0[:3], rv0[3:])
    if coe0[1] < 0 or coe0[1] > 1:
        raise ValueError('参数不符合要求')

    coe = J2Orbit(coe0, T)
    r, v = Orbit_Element_2_State_rv(coe, GM_Earth)
    rv = np.concatenate((r, v))
    return rv
