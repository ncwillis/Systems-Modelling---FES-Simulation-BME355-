import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import solve_ivp
import sim_regressions as reg
import external_state as ext


def get_state(t, x):
    U_stim = np.ones(10)
    U_stim = 400 * U_stim
    #u = get_excitation(U_stim, 30)

    u = reg.get_activation_regression()
    T_act = 0.01 #in seconds
    T_deact = 0.04 #in seconds
    B = 0.82 #viscosity param
    J = 0.0197 #inertia of food around ankle (kg m^2)
    d = 0.03 #d, the moment arm around ankle joint
    a = [2.10, -0.08, -7.97, 0.19, -1.79]
    T_ela = np.exp(a[0]+(a[1]*x[1])) - np.exp(a[2]+a[3]*x[1]) + a[4]
    x_ext = [ext.x_ext_1.eval(t), ext.x_ext_2.eval(t), ext.x_ext_3.eval(t), ext.x_ext_4.eval(t)]

    x1_dot = (u.eval(t) - x[0]) * ((u.eval(t) / T_act) - ((1 - u.eval(t)) / T_deact))
    x2_dot = x[2]
    x3_dot = (1/J) * F_m(x, x_ext) * d + T_grav(x[1]) + T_acc(x, x_ext) + T_ela + B * (x_ext[3] - x[2])

    return [x1_dot, x2_dot, x3_dot]


def F_m(x, x_ext):
    F_max = 600 #Newtons
    f_fl = 0.18
    f_fv = 0.023
    return x[0]*F_max*f_fl*(x_ext[2]-x[1])*f_fv*(x_ext[3] - x[2])


def T_grav(x_2):
    m_F = 1.025
    c_F = 11.45 #in cm
    g = 9.81 #m/s^2
    return - (m_F * c_F * math.cos(x_2) * g)


def T_acc(x, x_ext):
    m_F = 1.025 #in kg
    c_F = 11.45 #in cm
    return - m_F * c_F *(x_ext[0] * math.sin(x[1]) - x_ext[1]* math.cos(x[1]) )


def get_excitation(U_stim, freq_stim):
    U_tr = 22   #TODO: get threshold stimulation to elicit muscle response in mA
    U_sat = 46 #TODO: get stimulation value that elicits a full muscle response in mA (50 mA is on high end of range)
    fmax = 150 #maximum force produced by the tibialis
    fCF = 10    #TODO: force produced by tibialis at critical fusion frequency
    freq_CF = 5    #TODO: get critical fusion frequency (frequency reguired to elicit a sustained contraction)
    R = 15
    f_0 = R*np.log((((fmax/fCF)-1)*np.exp(freq_CF/R)-(fmax/fCF)))
    k = -150/(np.exp(f_0/R))

    excitation = []
    for i in range(len(U_stim)):
        excitation.append(((U_stim[i]-U_tr)/(U_sat-U_tr))*(((k-(fmax/fCF))/(1+(np.exp((freq_stim-f_0)/R))))+(fmax/fCF)))
    return excitation

U_sat = 46
U_stim = []
activation = reg.get_activation_regression()
time = np.arange(0, 0.4, 0.001)
for i in range(len(time)):
    U_stim.append(U_sat*activation.eval(time[i])[0])
print(U_stim)
exc = get_excitation(U_stim, 40)
print(exc)


#sol = solve_ivp(get_state(), [0, 0.4], [0, -15, -1.5], args=arg)
sol = solve_ivp(get_state, [0, 0.4], [0, -15, -1.5], rtol=1e-5, atol=1e-8)
time = sol.t
data1 = sol.y[0]
data2 = sol.y[1]
data3 = sol.y[2]
plt.figure()
plt.subplot(3, 1, 1)
plt.plot(time, data1)
plt.ylabel('Activation')
plt.subplot(3, 1, 2)
plt.plot(time, data2)
plt.ylabel('Ankle Angle')
plt.subplot(3,1,3)
plt.plot(time, data3)
plt.ylabel('Ankle Angular Velocity')
plt.xlabel('%Gait Cycle')
plt.show()