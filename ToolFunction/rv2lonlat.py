import numpy as np
from OrbitPredict.sofa.Mjday import Mjday
from OrbitPredict.sofa.ICRS2ITRS import ICRS2ITRS

def rv2lonlat(rv, date):
    # Make sure to define or import Mjday and ICRS2ITRS functions
    Mjd = Mjday(date[0], date[1], date[2], date[3], date[4], date[5])
    E = ICRS2ITRS(Mjd)
    rf = np.dot(E, rv[:3])  # Earth-fixed position

    lon = np.degrees(np.arctan2(rf[1], rf[0]))
    lat = np.degrees(np.arctan2(rf[2], np.linalg.norm(rf)))

    lonlat = np.array([lon, lat])
    return lonlat, E

# Make sure to define or import Mjday and ICRS2ITRS functions
