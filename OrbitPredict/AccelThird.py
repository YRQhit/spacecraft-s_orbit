import numpy as np
from OrbitPredict.JPL_Eph_DE405 import  JPL_Eph_DE405
import globals
def AccelThird(Pos, Mjd):
    r_Earth, r_moon, r_sun,  r_Mars = JPL_Eph_DE405(Mjd + 66.184 / 86400)
    r_moon = r_moon / 1000
    r_sun = r_sun / 1000

    erthToSun = (r_sun[0] ** 2 + r_sun[1] ** 2 + r_sun[2] ** 2) ** 1.5
    earthToMoon = (r_moon[0] ** 2 + r_moon[1] ** 2 + r_moon[2] ** 2) ** 1.5

    R_sun = ((Pos[0] - r_sun[0]) ** 2 + (Pos[1] - r_sun[1]) ** 2 + (Pos[2] - r_sun[2]) ** 2) ** 1.5
    R_moon = ((Pos[0] - r_moon[0]) ** 2 + (Pos[1] - r_moon[1]) ** 2 + (Pos[2] - r_moon[2]) ** 2) ** 1.5

    a_sun = -globals.GM_Sun * ((Pos - r_sun) / R_sun + r_sun / erthToSun)
    a_moon = -globals.GM_Moon * ((Pos - r_moon) / R_moon + r_moon / earthToMoon)

    a = a_sun + a_moon
    return a
