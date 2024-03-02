from poliastro.iod import izzo
from poliastro.bodies import Earth
from astropy import units as u
import numpy as np

k = Earth.k

# 追击策略
blue_max_push = 0.01

def lambert(escapee_rv, opponent_rv):
    r0 = np.array(opponent_rv[0:3])
    r = np.array(escapee_rv[0:3])
    v_org = np.array(opponent_rv[3:6])
    # 计算 tof
    tof = np.linalg.norm(r0 - r) / 60 * u.min

    # 使用 poliastro 的 izzo.lambert 计算 v0
    r0 = r0 * u.km
    r = r * u.km
    v0, v = izzo.lambert(k, r0, r, tof)
    v_push = v0.value - v_org
    # 如果 v0 的大小超过 blue_max_push，则按比例缩小至 blue_max_push
    v0_norm = np.linalg.norm(v_push)
    if v0_norm > blue_max_push:
        v_push = v_push / v0_norm * blue_max_push

    # 仅保留 x 和 y 分量
    v_push = v_push[0:2]

    return v_push

# red_rv = [4.117995088156707e+04, 8.753079979601422e+03, -72.362023354757740, -0.639743965382237, 3.009762576635815, 9.325566361179985e-04]
# blue_rv = [4.124450852427926e+04, 8.766802147740464e+03, -72.475465006572800, -0.639243092202387, 3.007406150574878, 9.318265118292234e-04]
#
# v0 = lambert(red_rv, blue_rv)
# print("Best v0:", v0)

def lamberthigh(r0, r, tf):
    # 计算 tof
    tof = tf / 60 * u.min

    # 使用 poliastro 的 izzo.lambert 计算 v0
    r0 = r0 * u.km
    r = r * u.km
    v0, v = izzo.lambert(k, r0, r, tof)

    return v0.value, v.value

# red_rv = [4.117995088156707e+04, 8.753079979601422e+03, -72.362023354757740]
# blue_rv = [4.124450852427926e+04, 8.766802147740464e+03, -72.475465006572800]
# tf = 100
# v0, v = lamberthigh(red_rv, blue_rv, tf)
# print(v0,v)
