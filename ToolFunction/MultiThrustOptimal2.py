import numpy as np
from scipy.optimize import minimize
from scipy.integrate import odeint
from OrbitPredict.OrbitPrediction import OrbitPrediction
from ToolFunction.fuelCost import fuelCost
from ToolFunction.Inertial2Orbit import Inertial2Orbit
from OrbitPredict.J2Cal_rvm import J2Cal_rvm
from ToolFunction.ThrustAngleCal2 import ThrustAngleCal2
#   多段脉冲优化函数  考虑航天器质量变化
#  输入 脉冲次数 Thrust_num
#      追踪星RV rv_c 第一次脉冲时刻 脉冲施加前RV
#       每次脉冲机动信息 Thrust_Impulsive 4*n 每列为一次脉冲的deltv 以及对应脉冲时刻
#       推力上限 Thrust_f
#       mass：航天器初始质量   k：质量变化率（假设任何脉冲推力恒定且大小相等）
#
#  输出 每次脉冲施加时间 t_total 1*n
#      每次脉冲施加角度（轨道系） Thrust_angle 2*n
def MultiThrustOptimal2(Thrust_num, rv_c, Thrust_Impulsive, Thrust_f, mass, k):
    t_total = np.zeros(Thrust_num)
    Thrust_angle = np.zeros((2, Thrust_num))
    E_rv = np.zeros((6, Thrust_num - 1))
    final_mass = mass

    for i in range(Thrust_num):
        dm = fuelCost(np.linalg.norm(Thrust_Impulsive[:3, i]), final_mass)
        final_mass -= dm

    cal_rv_c = rv_c
    for i in range(Thrust_num - 1):
        cal_rv_c += np.array([0, 0, 0, *Thrust_Impulsive[:3, i]])
        E_rv[:, i], _ = OrbitPrediction(cal_rv_c, Thrust_Impulsive[3, i + 1] - Thrust_Impulsive[3, i], 60, [1, 0],'RK4')
        cal_rv_c = E_rv[:, i]

    rv_calangel = E_rv[:, -1]
    E_rv[:, -1] = E_rv[:, -1] + np.array([0, 0, 0, *Thrust_Impulsive[:3, -1]])
    RVEnd = E_rv[:, -1]

    FinalTrip_t = Thrust_Impulsive[3, -1] - Thrust_Impulsive[3, -2]
    FinalImpulse_t = np.linalg.norm(Thrust_Impulsive[:3, -1]) / (Thrust_f / final_mass) * 1000
    FinalTrip_t -= FinalImpulse_t

    Tran = Inertial2Orbit(rv_calangel)
    Orbit_Impulse = np.dot(Tran, Thrust_Impulsive[:3, -1]) / np.linalg.norm(Thrust_Impulsive[:3, -1])
    FinalImpulse_Azimuth, FinalImpulse_Elevation = ThrustAngleCal2(Orbit_Impulse)
    E_rvm = np.hstack((E_rv[:6, -1], final_mass))
    RVm = odeint(lambda RVm, t: J2Cal_rvm(t, RVm, Thrust_f, [FinalImpulse_Azimuth, FinalImpulse_Elevation], k), E_rvm,
                 [FinalImpulse_t, 0])[-1, :6]
    E_rv[:, -1] = RVm

    t_total[-1] = FinalImpulse_t
    Thrust_angle[0, -1] = FinalImpulse_Azimuth
    Thrust_angle[1, -1] = FinalImpulse_Elevation

    A = None
    b = None
    Aeq = None
    beq = None
    pro_mass = mass

    for i in range(Thrust_num - 1):
        p = np.hstack(
            (rv_c, E_rv[:, i], pro_mass, Thrust_f, Thrust_Impulsive[3, i + 1] - Thrust_Impulsive[3, i], k))

        if i == Thrust_num - 2:
            p = np.hstack((rv_c, E_rv[:, i], pro_mass, Thrust_f, FinalTrip_t, k))

        Tran = Inertial2Orbit(rv_c)
        Azimuth, Elevation = ThrustAngleCal2(
            np.dot(Tran, Thrust_Impulsive[:3, i]) / np.linalg.norm(Thrust_Impulsive[:3, i]))
        x_0 = np.array([np.linalg.norm(Thrust_Impulsive[:3, i]) / (Thrust_f / pro_mass / 1000), Azimuth, Elevation])

        u_lb = np.array([0.5 * np.linalg.norm(Thrust_Impulsive[:3, i]) / (Thrust_f / pro_mass / 1000), -1, -90])
        u_ub = np.array([1.5 * np.linalg.norm(Thrust_Impulsive[:3, i]) / (Thrust_f / pro_mass / 1000), 360, 90])

        output = minimize(lambda x: CostFun(x, p), x0=x_0, bounds=[(lb, ub) for lb, ub in zip(u_lb, u_ub)], method='TNC')

        t_total[i] = output.x[0]
        pro_mass += k * output.x[0]
        Thrust_angle[0, i] = output.x[1]
        Thrust_angle[1, i] = output.x[2]
        _, _, new_rv = CostFun_(output.x, p)
        rv_c = new_rv

    u_lb = np.array([0.5 * t_total[-1], -1, -90])
    u_ub = np.array([1.5 * t_total[-1], 360, 90])
    x_0 = np.array([t_total[-1], FinalImpulse_Azimuth, FinalImpulse_Elevation])

    output = minimize(lambda x: CostFun4(x, rv_c, RVEnd, Thrust_f, x_0[0], pro_mass, k), x_0,
                      bounds=[(lb, ub) for lb, ub in zip(u_lb, u_ub)],method= 'TNC')

    t_total[-1] = output.x[0]
    pro_mass += k * output.x[0]
    Thrust_angle[0, -1] = output.x[1]
    Thrust_angle[1, -1] = output.x[2]

    return t_total, Thrust_angle


def CostFun4(x, rv, RV, Thrust_f, t0, m, k):
    t = x[0] - t0
    newrv, _ = OrbitPrediction(rv, -t, 1, [1, 0], 'RK4')
    newrvm = np.hstack((newrv, m))
    time_span = np.linspace(0, x[0], 100)  # 时间间隔可以根据问题调整
    rv2 = odeint(lambda RV, t: J2Cal_rvm(t, RV, Thrust_f, [x[1], x[2]], k), newrvm, time_span)
    finalrv = rv2[-1, :6]
    J = np.linalg.norm(RV - finalrv)
    return J


def CostFun(x, p):
    rv_c = p[0:6]
    E_rv = p[6:12]
    m = p[12]
    Thrust_f = p[13]
    T = p[14]
    k = p[15]
    rvm = np.concatenate((rv_c, [m]), axis=0)

    def differential_eqs(rvm, t):
        return J2Cal_rvm(t, rvm, Thrust_f, [x[1], x[2]], k)

    t_span = np.array([0, x[0]])
    RV = odeint(differential_eqs, rvm, t_span)  # 使用 odeint 进行数值积分

    rv = RV[-1, 0:6]
    rv, _ = OrbitPrediction(rv, T - x[0], 60, [1, 0], 'RK4')

    err = E_rv - rv
    minJ = np.linalg.norm(E_rv - rv)

    return minJ

def CostFun_(x, p):
    rv_c = p[0:6]
    E_rv = p[6:12]
    m = p[12]
    Thrust_f = p[13]
    T = p[14]
    k = p[15]
    rvm = np.concatenate((rv_c, [m]), axis=0)

    def differential_eqs(rvm, t):
        return J2Cal_rvm(t, rvm, Thrust_f, [x[1], x[2]], k)

    t_span = np.array([0, x[0]])
    RV = odeint(differential_eqs, rvm, t_span)  # 使用 odeint 进行数值积分

    rv = RV[-1, 0:6]
    rv, _ = OrbitPrediction(rv, T - x[0], 60, [1, 0], 'RK4')

    err = E_rv - rv
    minJ = np.linalg.norm(E_rv - rv)

    return minJ,err,rv