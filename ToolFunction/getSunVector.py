import numpy as np

def getSunVector(JD):
    Pi = 3.141592653589793238462643383279502884197169399375105
    T = (JD - 2451545.0) / 36525.
    lambdaM = 280.460 + 36000.771 * T
    M = 357.5277233 + 35999.05034 * T
    lambda_sun = lambdaM + 1.91466647 * np.sin(M * Pi / 180.) + 0.019994643 * np.sin(2 * M * Pi / 180.)
    r = 1.000140612 - 0.016708617 * np.cos(M * Pi / 180.) + 0.000139589 * np.cos(2 * M * Pi / 180.)
    e = 23.439291 - 0.0130042 * T

    sun_Vector_J2000 = np.zeros((3, 1))
    sun_Vector_J2000[0, 0] = r * np.cos(lambda_sun * Pi / 180.)
    sun_Vector_J2000[1, 0] = r * np.cos(e * Pi / 180) * np.sin(lambda_sun * Pi / 180.)
    sun_Vector_J2000[2, 0] = r * np.sin(e * Pi / 180.) * np.sin(lambda_sun * Pi / 180.)

    return sun_Vector_J2000

# 示例用法
# JD = 2459396.5  # 日期对应的Julian Date
# sun_Vector_J2000 = getSunVector(JD)
# print("Sun Vector (J2000):")
# print(sun_Vector_J2000)
