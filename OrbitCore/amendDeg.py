# % 角度修正函数，根据不同要求的角度范围调整实际角度值
# % 输入：deg0 - 原始角度      type - 角度范围
# % 输出：deg - 满足范围要求的角度值
# % 补充 - STK中相关角度的范围：
# %     轨道倾角："0 - 180"            近地点幅角："0 - 360"
# %     升交点赤经："-180 - 360"       真近点角："-180 - 360"
def amendDeg(deg0, type):
    if type == '':
        type = '-180 - 360'

    if type == '0 - 180':
        range_min = 0
        range_max = 180
    elif type == '0 - 360':
        range_min = 0
        range_max = 360
    elif type == '-180 - 360':
        range_min = -180
        range_max = 360
    else:
        print('该范围有误')
        return

    if deg0 > range_max:
        deg = deg0 % range_max
    elif deg0 > range_min:
        deg = deg0
    else:
        deg = range_max - abs(deg0) % range_max

    return deg
