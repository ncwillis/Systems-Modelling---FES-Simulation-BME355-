import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import solve_ivp
import sim_regressions as reg
import external_state as ext
import lookup_table as lut


def get_state(t, x):
    # Set input u
    u = exc

    #Variables
    T_act = 0.01 #in seconds
    T_deact = 0.04 #in seconds
    B = 0.82 #viscosity param
    J = 0.0197 #inertia of food around ankle (kg m^2)
    d = 0.03 #d, the moment arm around ankle joint
    a = [2.10, -0.08, -7.97, 0.19, -1.79]
    T_ela = np.exp(a[0]+(a[1]*x[1])) - np.exp(a[2]+a[3]*x[1]) + a[4]
    x_ext = [ext.x_ext_1.eval(t), ext.x_ext_2.eval(t), ext.x_ext_3.eval(t), ext.x_ext_4.eval(t)]

    #Get trajectory
    x1_dot = (u.eval(t) - x[0]) * ((u.eval(t) / T_act) - ((1 - u.eval(t)) / T_deact))
    x2_dot = x[2]
    x3_dot = (1 / J) * F_m(x, x_ext) * d + T_grav(x[1]) + T_acc(x, x_ext) + T_ela + B * (x_ext[3] - x[2])

    return [x1_dot, x2_dot, x3_dot]

def get_U_stim(max_current, timestep):
    time_exc = np.arange(0, 0.4 + timestep, timestep)
    U_stim = np.zeros((len(time_exc)))
    for i in range(len(time_exc)):
        U_stim[i] = reg.activation_reg.eval(time_exc[i])
    U_stim = U_stim/np.amax(U_stim)
    U_stim = U_stim*max_current
    for i in range(len(U_stim)):
        if U_stim[i] < 10:
            U_stim[i] = 10
        elif U_stim[i] > 50:
            U_stim[i] = 50;
    # print(U_stim)
    return time_exc, U_stim

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
    U_tr = 7   #get threshold stimulation to elicit muscle response in mA
    U_sat = 50 #get stimulation value that elicits a full muscle response in mA (50 mA is on high end of range)
    fmax = 150 #maximum force produced by the tibialis
    #Fmax should be 600
    fCF = 9    #force produced by tibialis at critical fusion frequency
    freq_CF = 5    #get critical fusion frequency (frequency reguired to elicit a sustained contraction)
    R = 15
    f_0 = R*np.log((((fmax/fCF)-1)*np.exp(freq_CF/R)-(fmax/fCF)))
    k = -150/(np.exp(f_0/R))

    excitation = []
    for i in range(len(U_stim)):
        if U_stim[i] < U_tr:
            excitation.append(0)
        elif U_stim[i] > U_sat:
            excitation.append(1)
        else:
            excitation.append(((U_stim[i]-U_tr)/(U_sat-U_tr))*(((k-(fmax/fCF))/(1+(np.exp((freq_stim-f_0)/R))))+(fmax/fCF)))
    return excitation

natural_excitation = np.array([
        [0.0011197287857882265, 0.0024038461538460343],
        [0.00336457614694724, 0],
        [0.006725109951707525, 0.0024038461538460343],
        [0.012330491117626824, 0.0024038461538460343],
        [0.011209414884442936, 0.0024038461538460343],
        [0.02018341453949643, -0.0024038461538460343],
        [0.016814796050362235, 0.0024038461538460343],
        [0.025783405915833085, 0.0024038461538460343],
        [0.030267710848568524, 0.0024038461538460343],
        [0.038115244480855515, 0.0024038461538460343],
        [0.04147308339082445, 0.007211538461538325],
        [0.0425860749396344, 0.014423076923076872],
        [0.04705960029320458, 0.024038461538461675],
        [0.05041743920317354, 0.028846153846153744],
        [0.053775278113142505, 0.033653846153846034],
        [0.058243413677130096, 0.048076923076922906],
        [0.061595862797516415, 0.05769230769230771],
        [0.06270615945153504, 0.06730769230769229],
        [0.06605591367713007, 0.07932692307692313],
        [0.06828728656433258, 0.08894230769230749],
        [0.07052135434632634, 0.09615384615384603],
        [0.07163434589513629, 0.10336538461538458],
        [0.07609978656433256, 0.12019230769230771],
        [0.07833115945153502, 0.1298076923076923],
        [0.07831768497757852, 0.1418269230769229],
        [0.08167282899275613, 0.14903846153846145],
        [0.08278312564677476, 0.15865384615384603],
        [0.08277504096240085, 0.16586538461538458],
        [0.08500102406002075, 0.18028846153846145],
        [0.08722700715764059, 0.19471153846153855],
        [0.08945568515005176, 0.20673076923076916],
        [0.09056598180407044, 0.21634615384615374],
        [0.09167358356329774, 0.22836538461538458],
        [0.0927865751121077, 0.23557692307692313],
        [0.09501794799931013, 0.2451923076923077],
        [0.09613093954812013, 0.25240384615384603],
        [0.09612554975853749, 0.2572115384615383],
        [0.09723045662297347, 0.2716346153846154],
        [0.09946182951017596, 0.28125],
        [0.10057212616419464, 0.2908653846153846],
        [0.10168242281821321, 0.3004807692307693],
        [0.10503217704380818, 0.3124999999999999],
        [0.10501061788547772, 0.33173076923076916],
        [0.1083603721110728, 0.34375],
        [0.10835228742669889, 0.35096153846153844],
        [0.10946258408071746, 0.360576923076923],
        [0.10946258408071746, 0.360576923076923],
        [0.11393071964470508, 0.3749999999999999],
        [0.11391994006553985, 0.3846153846153846],
        [0.1139145502759572, 0.38942307692307687],
        [0.11838268583994482, 0.40384615384615385],
        [0.11949567738875477, 0.4110576923076923],
        [0.12060866893756472, 0.4182692307692307],
        [0.12284273671955848, 0.42548076923076916],
        [0.1250714147119697, 0.4375],
        [0.12505524534322182, 0.45192307692307687],
        [0.1261709317868231, 0.45673076923076916],
        [0.12616554199724045, 0.46153846153846145],
        [0.12840230467402558, 0.46634615384615385],
        [0.12951260132804415, 0.4759615384615383],
        [0.1317520588996206, 0.4783653846153846],
        [0.13398612668161436, 0.485576923076923],
        [0.1362174995688169, 0.4951923076923076],
        [0.13845426224560192, 0.5],
        [0.14293048249396345, 0.5072115384615384],
        [0.14292509270438086, 0.5120192307692307],
        [0.14516455027595726, 0.5144230769230769],
        [0.14964616031390138, 0.516826923076923],
        [0.1530066941186617, 0.5192307692307693],
        [0.16309368532252505, 0.5216346153846153],
        [0.16309368532252505, 0.5216346153846153],
        [0.1675860749396344, 0.5144230769230769],
        [0.17320223568471893, 0.5048076923076922],
        [0.17656546438427045, 0.5048076923076922],
        [0.17769193040703696, 0.5],
        [0.17993947266298727, 0.4951923076923076],
        [0.18106324379096245, 0.49278846153846156],
        [0.1855556334080718, 0.485576923076923],
        [0.1866847943256295, 0.4783653846153846],
        [0.18893233658157993, 0.4735576923076923],
        [0.1900614974991377, 0.46634615384615385],
        [0.19118526862711283, 0.4639423076923077],
        [0.19568304803380476, 0.45192307692307687],
        [0.19905436141773025, 0.44471153846153844],
        [0.20018352233528808, 0.4375],
        [0.20243645438082103, 0.4278846153846153],
        [0.2069315388927217, 0.4182692307692307],
        [0.211423928509831, 0.4110576923076923],
        [0.21591362323214902, 0.4062499999999999],
        [0.22377193644360122, 0.3966346153846154],
        [0.22040601284925843, 0.39903846153846145],
        [0.22266163978958264, 0.3870192307692307],
        [0.22939887676785103, 0.37740384615384615],
        [0.23388857149016906, 0.37259615384615374],
        [0.23613341885132805, 0.3701923076923077],
        [0.24398903716798898, 0.36298076923076916],
        [0.24847334210072441, 0.36298076923076916],
        [0.2518392656950673, 0.360576923076923],
        [0.2552051892894102, 0.35817307692307687],
        [0.2596921891169369, 0.3557692307692307],
        [0.26530296007243875, 0.35096153846153844],
        [0.2709083412383581, 0.35096153846153844],
        [0.2753953410658848, 0.3485576923076923],
        [0.28100341712659543, 0.34615384615384603],
        [0.2843693407209383, 0.34375],
        [0.2899747218868575, 0.34375],
        [0.3000671028803036, 0.34134615384615385],
        [0.30455410270783034, 0.3389423076923076],
        [0.30455410270783034, 0.3389423076923076],
        [0.30680164496378065, 0.3341346153846154],
        [0.3135307972576751, 0.33173076923076916],
        [0.3168940259572267, 0.33173076923076916],
        [0.321386415574336, 0.3245192307692307],
        [0.32363126293549505, 0.3221153846153846],
        [0.32475772895826155, 0.3173076923076923],
        [0.3292474236805795, 0.3124999999999999],
        [0.33486088953087273, 0.30528846153846145],
        [0.3371084317868231, 0.3004807692307693],
        [0.33935597404277335, 0.29567307692307687],
        [0.34160621119351514, 0.28846153846153844],
        [0.3427326772162815, 0.28365384615384603],
        [0.3461066854949983, 0.27403846153846145],
        [0.3483596175405312, 0.26442307692307687],
        [0.3517309309244568, 0.2572115384615383],
        [0.35286009184201456, 0.25],
        [0.35623140522594005, 0.24278846153846145],
        [0.35736326103828914, 0.23317307692307687],
        [0.3607345744222145, 0.22596153846153832],
        [0.3629848115729563, 0.21875],
        [0.3641139724905141, 0.21153846153846145],
        [0.3652404385132804, 0.20673076923076916],
        [0.3686117518972059, 0.19951923076923062],
        [0.36973552302518115, 0.19711538461538458],
        [0.37310953130389796, 0.18749999999999978],
        [0.3753678531390135, 0.1730769230769229],
        [0.3776207851845464, 0.16346153846153855],
        [0.3787499461021043, 0.15624999999999978],
        [0.37988180191445325, 0.1466346153846152],
        [0.38326389487754403, 0.1298076923076923],
        [0.38663790315626084, 0.12019230769230771],
        [0.3877670640738186, 0.11298076923076916],
        [0.3911383774577441, 0.10576923076923084],
        [0.39451508063125224, 0.09375],
        [0.3967653177819939, 0.08653846153846145],
        [0.3990182498275267, 0.07692307692307687],
        [0.40127387676785103, 0.06490384615384626],
        [0.40239764789582605, 0.06249999999999978],
        [0.4046451901517765, 0.05769230769230771],
        [0.4068954273025182, 0.050480769230769384],
        [0.4102667406864436, 0.043269230769230616],
        [0.4125196727319766, 0.033653846153846034],
        [0.4158936810106934, 0.024038461538461675],
        [0.42150984175577794, 0.014423076923076872],
        [0.42375199422214566, 0.014423076923076872],
        [0.425999536478096, 0.00961538461538436]
    ])

pulse_times = np.linspace(0,0.4,14)
emg_vals = np.zeros(len(pulse_times))
angle_vals = np.zeros(len(pulse_times))
pulse_train_unique = np.zeros(len(pulse_times))
# Make rectangular pulse shape with max 250 us
pulse_train_rectangular = np.ones(len(pulse_times))
pulse_train_rectangular = pulse_train_rectangular*250
# Make triangular pulse shape with max val of 350 us
pulse_train_triangular = [100., 100., 150., 200., 250., 300., 350., 350., 300., 250., 200., 150., 100., 100.]
pulse_train_trapezoidal = [100., 100., 200., 300., 300., 300., 300., 300., 300., 300., 300., 200., 100., 100]

# Get corresponding desired EMG vals and angle vals for pulse times
# Get optimal pulse width
excitation_unique = reg.get_excitation_regression(natural_excitation)
angles = reg.get_angle_regression()
for i in range(len(pulse_times)):
    emg_vals[i] = excitation_unique.eval(pulse_times[i])
    angle_vals[i] = angles.eval(pulse_times[i])
    #get pulse width from lookup table by passing EMG amplitude and angle
    pulse_train_unique[i] = lut.lookup_table(np.round(angle_vals[i]), emg_vals[i])

excitation_rectangular_vals = []
excitation_triangular_vals = []
excitation_trapezoidal_vals = []

for i in range(len(pulse_times)):
    excitation_rectangular_vals.append([pulse_times[i], lut.lookup_table(np.round(angle_vals[i]), pulse_train_rectangular[i])])
    excitation_triangular_vals.append([pulse_times[i], lut.lookup_table(np.round(angle_vals[i]), pulse_train_triangular[i])])
    excitation_trapezoidal_vals.append([pulse_times[i], lut.lookup_table(np.round(angle_vals[i]), pulse_train_trapezoidal[i])])
excitation_rectangular_vals = np.array(excitation_rectangular_vals)
excitation_triangular_vals = np.array(excitation_triangular_vals)
excitation_trapezoidal_vals = np.array(excitation_trapezoidal_vals)

excitation_rectangular = reg.get_excitation_regression(excitation_rectangular_vals)
excitation_triangular = reg.get_excitation_regression(excitation_triangular_vals)
excitation_trapezoidal = reg.get_excitation_regression(excitation_trapezoidal_vals)

# Perform numerical integration
exc = excitation_unique
sim_unique = solve_ivp(get_state, [0, 0.4], [0, -15, 0], rtol=1e-5, atol=1e-8)

exc = excitation_rectangular
sim_rectangular = solve_ivp(get_state, [0, 0.4], [0, -15, 0], rtol=1e-5, atol=1e-8)

exc = excitation_triangular
sim_triangular = solve_ivp(get_state, [0, 0.4], [0, -15, 0], rtol=1e-5, atol=1e-8)

exc = excitation_trapezoidal
sim_trapezoidal = solve_ivp(get_state, [0, 0.4], [0, -15, 0], rtol=1e-5, atol=1e-8)
