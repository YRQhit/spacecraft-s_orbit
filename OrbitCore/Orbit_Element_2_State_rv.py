import numpy as np
# %---------coe（classical orbit elements）-------------%
# %a：轨道半长轴
# %e：轨道偏心率
# %incl：轨道倾角(单位/deg)
# %RAAN：升交点赤经(单位/deg)
# %omegaa：近地点幅角(单位/deg)
# %TA：真近点角(单位/deg)
def Orbit_Element_2_State_rv(coe, muu=3.986004415e+05):
    d_2_r = np.pi / 180
    a = coe[0]
    e = coe[1]
    incl = coe[2] * d_2_r
    RAAN = coe[3] * d_2_r
    omegaa = coe[4] * d_2_r
    TA = coe[5] * d_2_r

    h = np.sqrt(a * muu * (1 - e ** 2))

    rp = (h ** 2 / muu) * (1 / (1 + e * np.cos(TA))) * (np.cos(TA) * np.array([1, 0, 0]) + np.sin(TA) * np.array([0, 1, 0]))
    vp = (muu / h) * (-np.sin(TA) * np.array([1, 0, 0]) + (e + np.cos(TA)) * np.array([0, 1, 0]))

    R3_RAAN = np.array([[np.cos(RAAN), np.sin(RAAN), 0],
                        [-np.sin(RAAN), np.cos(RAAN), 0],
                        [0, 0, 1]])

    R1_incl = np.array([[1, 0, 0],
                        [0, np.cos(incl), np.sin(incl)],
                        [0, -np.sin(incl), np.cos(incl)]])

    R3_omegaa = np.array([[np.cos(omegaa), np.sin(omegaa), 0],
                          [-np.sin(omegaa), np.cos(omegaa), 0],
                          [0, 0, 1]])

    Q_px = np.dot(np.dot(R3_RAAN.T, R1_incl.T), R3_omegaa.T)

    r = np.dot(Q_px, rp)
    v = np.dot(Q_px, vp)

    return r, v
