import numpy as np
from OrbitPredict.Geodetic import Geodetic
from ToolFunction.rv2lonlat import rv2lonlat
from OrbitPredict.AddTime import AddTime
def lonlattablegen(rvdata, date):
    lonlattable = []

    for i in range(len(rvdata[0])):
        lonlat_record, E = rv2lonlat(rvdata[:3, i], AddTime(date, 60 * i))
        lon, lat, _ = Geodetic(np.dot(E, rvdata[:3, i]))
        lonlat_T_record = [lon, lat, 60 * i]
        lonlattable.append(lonlat_T_record)

        if i != 0:
            if lonlattable[0][i] - lonlattable[0][i - 1] > 90:
                lonlattable[0][i] = lonlat_record[0] - 180
            elif lonlattable[0][i] - lonlattable[0][i - 1] < -90:
                lonlattable[0][i] = lonlat_record[0] + 180

        if lonlattable[0][i] > 180:
            lonlattable[0][i] -= 360
        elif lonlattable[0][i] < -180:
            lonlattable[0][i] += 360

    nodel = 1
    return lonlattable

# Make sure to define or import rv2lonlat, AddTime, and Geodetic functions
