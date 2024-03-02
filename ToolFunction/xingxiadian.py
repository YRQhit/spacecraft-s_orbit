from skyfield.api import Topos, load
import numpy as np

# 计算星下点
from ToolFunction.rv2lonlat import rv2lonlat
from OrbitPredict.AddTime import AddTime
from OrbitPredict.Geodetic import Geodetic
# 输入惯性系下的rv,时间，输出经纬度
# 例子
# rv = [ 6.20489388e+03  7.08819884e+02  2.88195974e+03 -3.01287254e+00  -1.44618334e+00  6.84402314e+00]
# date = [2018,1,1,0,0,0]

def xingxiadian(rv,date):
# 使用自定义函数 rv2latlon 计算星下点经纬度
    # 使用自定义函数 rv2lonlat 计算星下点经纬度
    lonlat_c, E2 = rv2lonlat(rv, date)
    _, lat, _ = Geodetic(np.dot(E2, rv[:3]))
    lon = lonlat_c[0]
    lat = lat
    return lon,lat

