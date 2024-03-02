#  多段脉冲优化函数  考虑航天器质量变化
#  输入 脉冲次数 Thrust_num
#       追踪星RV rv_c 第一次脉冲时刻 脉冲施加前RV
#       每次脉冲机动信息 Thrust_Impulsive 4*n 每列为一次脉冲的deltv 以及对应脉冲时刻
#       推力上限 Thrust_f
#       mass：航天器初始质量   k：质量变化率（假设任何脉冲推力恒定且大小相等）
#
#  输出 每次脉冲施加时间 t_total 1*n
#       每次脉冲施加角度（轨道系） Thrust_angle 2*n
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
from OrbitPredict.J2OrbitRV import J2OrbitRV
from ToolFunction.Inertial2Orbit import Inertial2Orbit
from ToolFunction.ThrustAngleCal2 import ThrustAngleCal2
from OrbitPredict.TwoBodyCal_rvm import TwoBodyCal_rvm
from OrbitPredict.TwoBodyCal import TwoBodyCal
def MultiThrustOptimal(Thrust_num, rv_c, Thrust_Impulsive, Thrust_f):
    t_total = np.zeros(Thrust_num)
    Thrust_angle = np.zeros((2, Thrust_num))
    E_rv = np.zeros((6, Thrust_num-1))  # 每列为每次脉冲时刻的期望RV（施加脉冲前）

    # 计算期望RV
    cal_rv_c = rv_c.copy()
    for i in range(Thrust_num-1):
        cal_rv_c = cal_rv_c + np.array([0, 0, 0, *Thrust_Impulsive[0:3, i]])  # 计算施加脉冲后的RV
        E_rv[:, i] = J2OrbitRV(cal_rv_c, Thrust_Impulsive[3, i+1] - Thrust_Impulsive[3, i])  # J2适用
        cal_rv_c = E_rv[:, i]

    # 最终段期望RV修正
    rv_calangel = E_rv[:, -1]
    E_rv[:, -1] = E_rv[:, -1] + np.array([0, 0, 0, *Thrust_Impulsive[0:3, -1]])
    RVEnd = E_rv[:, -1]

    FinalTrip_t = Thrust_Impulsive[3, Thrust_num - 1] - Thrust_Impulsive[3, Thrust_num - 2]  # 最终段转移时间
    FinalImpulse_t = np.linalg.norm(Thrust_Impulsive[0:3, -1]) / Thrust_f  # 最末次脉冲时间
    FinalTrip_t = FinalTrip_t - FinalImpulse_t
    Tran = Inertial2Orbit(rv_calangel)
    Orbit_Impulse = np.dot(Tran, Thrust_Impulsive[0:3, -1]) / np.linalg.norm(Thrust_Impulsive[0:3, -1])  # 轨道系脉冲矢量
    FinalImpulse_Azimuth, FinalImpulse_Elevation = ThrustAngleCal2(Orbit_Impulse)  # 轨道系脉冲角度

    RV, _ = odeint(TwoBodyCal_rvm, E_rv[0:6, -1], [FinalImpulse_t, 0], args=(Thrust_f, [FinalImpulse_Azimuth, FinalImpulse_Elevation]), full_output=True)

    E_rv[:, -1] = RV[-1, :]

    t_total[-1] = FinalImpulse_t
    Thrust_angle[0, -1] = FinalImpulse_Azimuth
    Thrust_angle[1, -1] = FinalImpulse_Elevation

    # 逐段修正脉冲
    A = np.empty((0, 3))
    b = np.empty((0,))
    Aeq = np.empty((0, 3))
    beq = np.empty((0,))
    for i in range(Thrust_num - 1):
        p = np.concatenate((rv_c, E_rv[:, i], [Thrust_f, Thrust_Impulsive[3, i+1] - Thrust_Impulsive[3, i]]))

        # 最终段需要用修正后的转移时间
        if i == Thrust_num - 2:
            p = np.concatenate((rv_c, E_rv[:, i], [Thrust_f, FinalTrip_t]))

        Tran = Inertial2Orbit(rv_c)
        Azimuth, Elevation = ThrustAngleCal2(np.dot(Tran, Thrust_Impulsive[0:3, i]) / np.linalg.norm(Thrust_Impulsive[0:3, i]))
        x_0 = np.array([np.linalg.norm(Thrust_Impulsive[0:3, i]) / Thrust_f, Azimuth, Elevation])

        u_lb = np.array([0.5 * np.linalg.norm(Thrust_Impulsive[0:3, i]) / Thrust_f, -1, -90])
        u_ub = np.array([1.5 * np.linalg.norm(Thrust_Impulsive[0:3, i]) / Thrust_f, 360, 90])

        def CostFun(x, p):
            _, _, new_rv = CostFun(x, p)
            return np.linalg.norm((E_rv[:, i+1] - new_rv)[0:3])

        output = minimize(lambda x: CostFun(x, p), x0=x_0, bounds=[(lb, ub) for lb, ub in zip(u_lb, u_ub)])
        t_total[i] = output.x[0]
        Thrust_angle[:, i] = output.x[1:]
        _, _, new_rv = CostFun(output.x, p)
        rv_c = new_rv

    # 最后一个脉冲修正
    u_lb = np.array([0.5 * t_total[-1], -1, -90])
    u_ub = np.array([1.5 * t_total[-1], 360, 90])
    x_0 = np.array([t_total[-1], FinalImpulse_Azimuth, FinalImpulse_Elevation])

    output = minimize(lambda x: CostFun2(x, rv_c.T, RVEnd.T, Thrust_f, x_0[0]), x_0, bounds=[(lb, ub) for lb, ub in zip(u_lb, u_ub)])
    t_total[0, -1] = output.x[0]

    output = minimize(lambda x: CostFun3(x, rv_c.T, RVEnd.T, Thrust_f, x_0[0]), x_0[1:], bounds=[(lb, ub) for lb, ub in zip(u_lb[1:], u_ub[1:])])
    Thrust_angle[0, -1] = output.x[0]
    Thrust_angle[1, -1] = output.x[1]
    return t_total,Thrust_angle

def CostFun2(x, rv1, rv2, Thrust_f, t0):
    def dynamics(t, RV, Thrust_f, deg):
        return TwoBodyCal(t, RV, Thrust_f, deg)

    t_range = [x[1], 0] if x[1] > t0 else [-x[1], 0]

    RV, _ = odeint(dynamics, rv2, t_range, args=(Thrust_f, [x[2], x[3]]), full_output=True)
    t = x[1] - t0
    if t > 0:
        temp, _ = odeint(dynamics, rv1, [0, t], args=(0, [x[2], x[3]]), full_output=True)
        rv = temp[-1]
    elif t < 0:
        temp, _ = odeint(dynamics, rv1, [-t, 0], args=(0, [x[2], x[3]]), full_output=True)
        rv = temp[-1]
    else:
        rv = rv1

    J = np.linalg.norm((RV[-1] - rv))
    return J
def CostFun3(x, rv, RV, Thrust_f, t0):
    data, _ = odeint(TwoBodyCal, rv, [0, t0], args=(Thrust_f, [x[0], x[1]]), full_output=True)
    posvel = J2OrbitRV(data[-1], 86400 - t0)

    rv2 = J2OrbitRV(RV, 86400 - t0)
    J = np.linalg.norm(rv2 - posvel)
    return J