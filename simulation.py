import numpy as np
import math
from scipy.integrate import solve_ivp
import sim_regressions as reg
import external_state as ext
import lookup_table as lut
import csv


def get_state(t, x):

    """
    State equations obtained from research from Benoussaad et. al. from paper:
    "Nonlinear Model Predictive Control of Joint Ankle by Electrical Stimulation For Drop Foot Correction"
    M. Benoussaad, K. Mombaur and C. Azevedo-Coste, "Nonlinear Model Predictive Control of Joint Ankle by Electrical
    Stimulation For Drop Foot Correction", in IEEE International Conference on Intelligent Robots and Systems, Tokyo,
    2013, pp. 983-989.
    """

    """
    :param x: state vector (muscle activation, ankle angle, ankle angular velocity)
    :param t: current time
    :return: derivative of state vector
    """

    u = exc     #input muscle excitation (global variable)
    T_act = 0.01    #activation time constant (s)
    T_deact = 0.04  #deactivation time constant (s)
    B = 0.82        #viscosity parameter
    J = 0.0197      #inertia of food around ankle (kg m^2)
    d = 0.037       #d, the moment arm around ankle joint
    x_ext = [ext.x_ext_1.eval(t), ext.x_ext_2.eval(t), ext.x_ext_3.eval(t), ext.x_ext_4.eval(t)] #external state vector

    x1_dot = (u.eval(t) - x[0]) * ((u.eval(t) / T_act) - ((1 - u.eval(t)) / T_deact))
    x2_dot = x[2]
    x3_dot = (1 / J) * ((F_m(x, x_ext) * d) + T_grav(x[1]) + T_acc(x, x_ext)  + (B * (x_ext[3] - x[2])))

    return [x1_dot, x2_dot, x3_dot]

def F_m(x, x_ext):
    """
    :param x: state vector (muscle activation, ankle angle, ankle angular velocity)
    :param x_ext: external state vector (linear accelerations of ankle in horizonal and vertical, rotation velocity of shank)
    :return: TA muscular force
    """

    F_max = 600     # Maximum isometric force
    d = 0.037       # Moment arm of TA wrt ankle
    l_t = 0.223     # Constant tendon length
    l_mt = 0.321 + d*(x_ext[2]-x[1])    # muscle tendon length
    l_ce = l_mt - l_t   # contractile element length
    v_ce = d*(x_ext[3]-x[2])    # contractile velocity
    l_ce_opt = 0.1      # optimal length of contractile element for max force
    W = 0.56            # shape parameter for force-length curve
    v_max = -0.9        # maximum contractile speed (shortening)
    av = 1.33           # first force-velocity parameter
    fv1 = 0.18          # second force-velocity parameter
    fv2 = 0.023         # third force-velocity parameter
    f_fl_exp = ((l_ce-l_ce_opt)/(W*l_ce_opt))**2
    f_fl = np.exp(-f_fl_exp)
    if v_ce < 0:
        f_fv = (1-(v_ce/v_max))/(1+(v_ce/(v_max*fv1)))
    else:
        f_fv =(1+(av*(v_ce/fv2)))/(1+(v_ce/fv2))

    return x[0]*F_max*f_fl*f_fv


def T_grav(x_2):
    """
    :param x_2: ankle angular position
    :return:  gravity torque of the foot around the ankle
    """
    m_F = 1.025     # mass of foot (kg)
    c_F = 0.1145    # center of mass location wrt ankle (cm)
    g = 9.81        # acceleration due to gravity (m/s^2)
    return - (m_F * c_F * math.cos(math.radians(x_2)) * g)


def T_acc(x, x_ext):
    """
    :param x: state vector (muscle activation, ankle angle, ankle angular velocity)
    :param x_ext: external state vector (linear accelerations of ankle in horizonal and vertical, rotation velocity of shank)
    :return: torque induced by movement of ankle
    """
    m_F = 1.025     # mass of foot (kg)
    c_F = 0.1145    # center of mass location wrt ankle (cm)
    return m_F * c_F *(x_ext[0] * math.sin(math.radians(x[1])) - x_ext[1]* math.cos(math.radians(x[1])))

def parse_data(file_name):
    """
    :param file_name: Name of text file for given data
    :return: parsed data (time and dependent var.)
    """
    data = []
    with open(file_name) as csv_file:
        csv_read = csv.reader(csv_file, delimiter=',')
        for row in csv_read:
            row_content = [float(row[0]), float(row[1])]
            data.append(row_content)
    data = np.array(data)
    return data

def get_angle_vals(pulse_times):
    """
    :param pulse_times: Array of discrete time points when a change in pulse-width is permitted
    :return: Array of natural ankle angles at each time point during dorsiflexion
    """
    angle_vals = np.zeros(len(pulse_times))
    angles = reg.get_angle_regression()
    for i in range(len(pulse_times)):
        angle_vals[i] = angles.eval(pulse_times[i])
    return angle_vals

def construct_pulse_trains():
    """
    :return: Arrays pulse-widths various signal shapes
    """
    emg_vals = np.zeros(len(pulse_times))
    pulse_train_unique = np.zeros(len(pulse_times))

    pulse_train_rectangular = np.ones(len(pulse_times))
    pulse_train_rectangular = pulse_train_rectangular*250
    pulse_train_triangular = [100., 100., 150., 200., 250., 300., 350., 350., 300., 250., 200., 150., 100., 100.]
    pulse_train_trapezoidal = [100., 100., 200., 300., 300., 300., 300., 300., 300., 300., 300., 200., 100., 100]

    angle_vals = get_angle_vals(pulse_times)
    for i in range(len(pulse_times)):
        emg_vals[i] = excitation_unique.eval(pulse_times[i])
        pulse_train_unique[i] = lut.lookup_table(np.round(angle_vals[i]), emg_vals[i])

    return pulse_train_unique, pulse_train_rectangular, pulse_train_triangular, pulse_train_trapezoidal

def get_excitation_vals():
    """
    :return: Array of muscle excitations elicited by each signal shape
    """
    excitation_rectangular_vals = []
    excitation_triangular_vals = []
    excitation_trapezoidal_vals = []
    angle_vals = get_angle_vals(pulse_times)
    for i in range(len(pulse_times)):
        excitation_rectangular_vals.append([pulse_times[i], lut.lookup_table(np.round(angle_vals[i]), pulse_train_rectangular[i])])
        excitation_triangular_vals.append([pulse_times[i], lut.lookup_table(np.round(angle_vals[i]), pulse_train_triangular[i])])
        excitation_trapezoidal_vals.append([pulse_times[i], lut.lookup_table(np.round(angle_vals[i]), pulse_train_trapezoidal[i])])
    excitation_rectangular_vals = np.array(excitation_rectangular_vals)
    excitation_triangular_vals = np.array(excitation_triangular_vals)
    excitation_trapezoidal_vals = np.array(excitation_trapezoidal_vals)

    return excitation_rectangular_vals, excitation_triangular_vals, excitation_trapezoidal_vals

natural_excitation = parse_data('EMG_data.txt')

excitation_unique = reg.get_excitation_regression(natural_excitation)
pulse_times = np.linspace(0,0.4,14)
pulse_train_unique, pulse_train_rectangular, pulse_train_triangular, pulse_train_trapezoidal = construct_pulse_trains()

excitation_rectangular_vals, excitation_triangular_vals, excitation_trapezoidal_vals = get_excitation_vals()

excitation_rectangular = reg.get_excitation_regression(excitation_rectangular_vals)
excitation_triangular = reg.get_excitation_regression(excitation_triangular_vals)
excitation_trapezoidal = reg.get_excitation_regression(excitation_trapezoidal_vals)

# Perform numerical integration
exc = excitation_unique
sim_unique = solve_ivp(get_state, [0, 0.4], [0, -15, 10], rtol=1e-5, atol=1e-8)

exc = excitation_rectangular
sim_rectangular = solve_ivp(get_state, [0, 0.4], [0, -15, 10], rtol=1e-5, atol=1e-8)

exc = excitation_triangular
sim_triangular = solve_ivp(get_state, [0, 0.4], [0, -15, 10], rtol=1e-5, atol=1e-8)

exc = excitation_trapezoidal
sim_trapezoidal = solve_ivp(get_state, [0, 0.4], [0, -15, 10], rtol=1e-5, atol=1e-8)
