# 计算大气密度（NRLMSISE-00模型）
# 输入：PosVel - 卫星位置速度        startTime - 初始时刻（年月日时分秒）    addMjd - 增加儒略日
# 输出：dens - 大气密度
import numpy as np
import pysat
from datetime import datetime
from OrbitPredict.AddTime import AddTime
from OrbitPredict.Geodetic import Geodetic
from nrlmsise00 import msise_model
import globals
def ComputeDensity_HPOP(PosVel, startTime, addMjd):
    lon, lat, height = Geodetic(np.dot(globals.E, PosVel[:3]))

    if height >= 1000000:  # Default atmospheric density is 0 above 1000 km
        dens = 0
        return dens
    else:
        startTime = AddTime(startTime, addMjd * 86400)

    # Calculate the day of the year
    year = startTime[0]
    month = startTime[1]
    day = startTime[2]
    hour= startTime[3]
    minute= startTime[4]
    second= startTime[5]

    start_time_indices = [i for i, row in enumerate(globals.spaceWheather) if
                          row[0] == startTime[0] and row[1] == startTime[1] and row[2] == startTime[2]]

    num = start_time_indices[0]  # Use the first matching index

    # Extract values from spaceWheather
    ap = globals.spaceWheather[num][ 3]
    f107A = globals.spaceWheather[num][ 5]
    f107 = globals.spaceWheather[num - 1][ 4]

    # Create a datetime object for the start time
    start_time = datetime(year, month, day, hour, minute, second)

    height = height * 0.001#从m转化为km
    f107A = f107A
    f107 = f107
    #花了两天，msise_model替换atmosnrlmsise00
    #测试例子
    #Matlab：[~,rho] = atmosnrlmsise00(400000, 60, -70, startTime(1), dayOfYear, second,150, 150,4)
    #python：msise_model(datetime(2009, 6, 21, 8, 3, 20), 400, 60, -70, 150, 150, 4)
    rho,_ = msise_model(start_time,height,lat,lon,f107A,f107,ap)
    dens = rho[5]*1000
    return dens
