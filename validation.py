import matplotlib.pyplot as plt
import numpy as np
import simulation as sim
from sklearn.metrics import mean_squared_error

t_exc = np.linspace(0,0.4,40)

exc_test = np.zeros(len(t_exc))
for i in range(len(t_exc)):
    exc_test[i] = sim.excitation_test.eval(t_exc[i])

sim_test = sim.sim_test

# Plot test
paper_angle = np.array([
        [0.0004384197090873121, -14.941520467836256],
        [0.013496727173300714, -14.898635477582847],
        [0.027017025298702896, -13.869395711500974],
        [0.040054590325067, -12.883040935672515],
        [0.053102526570355746, -12.368421052631579],
        [0.06664450997169129, -12.325536062378166],
        [0.08164600672714972, -12.66861598440546],
        [0.09327025746550982, -13.397660818713451],
        [0.1063445931771522, -14.08382066276803],
        [0.12038345224878669, -14.641325536062379],
        [0.13344835957958848, -14.898635477582847],
        [0.14747213324187802, -14.769980506822613],
        [0.16052289800141897, -14.38401559454191],
        [0.17404885315532548, -13.612085769980506],
        [0.18612378049786563, -12.840155945419102],
        [0.200128697398474, -11.853801169590643],
        [0.21316437674866995, -10.781676413255358],
        [0.22668373203598813, -9.709551656920077],
        [0.23971846854810008, -8.594541910331383],
        [0.2532359381592501, -7.436647173489279],
        [0.2667496364180638, -6.107212475633528],
        [0.2797815444159236, -4.863547758284598],
        [0.29329524267473733, -3.5341130604288473],
        [0.305844417573559, -2.333333333333332],
        [0.3203273533827851, -1.0896686159844045],
        [0.3333573757044768, 0.23976608187134651],
        [0.3468654169347862, 1.8265107212475646],
        [0.35941364899552386, 3.0701754385964932],
        [0.37293866131134634, 3.884990253411307],
        [0.3864646164652529, 4.6569200779727105],
        [0.3990345338019238, 4.9142300194931785]
    ])
paper_angle_time = paper_angle[:,0]
paper_angle_vals = paper_angle[:,1]

paper_activation = np.array([
    [-0.0006060632222361292, -0.0034188181767166315],
    [0.012670347364013478, 0.6529347014361627],
    [0.02617640583789104, 0.48193681229738544],
    [0.03963411738805745, 0.35880237763264117],
    [0.05366853867106395, 0.26472530745621803],
    [0.06585714347381283, 0.1980065527348388],
    [0.07984149115726136, 0.15350234612080693],
    [0.09319905550831181, 0.12951363858084508],
    [0.10714886967482379, 0.11919761373397975],
    [0.12109177713794847, 0.11571922524054756],
    [0.13806154736056017, 0.11564670485498085],
    [0.15563565412956115, 0.11728100354400228],
    [0.1744218873430344, 0.11891012220548303],
    [0.19562892009375854, 0.1239476989885997],
    [0.21683595284448265, 0.12898527577171615],
    [0.2325815098916943, 0.14088379903219816],
    [0.24650024389296343, 0.1613371377757825],
    [0.25860078822752414, 0.18179824656067756],
    [0.27009699601569553, 0.20055253627098457],
    [0.28220099370194984, 0.21759482687916298],
    [0.2949145079621341, 0.23121570929685442],
    [0.30762629554647136, 0.24654600080290423],
    [0.320344989834196, 0.2550386559555209],
    [0.3328547563444546, 0.47036981079949414],
    [0.34569949796899757, 0.3540756025019533],
    [0.3590881424852909, 0.2993175313715418],
    [0.3728894625289758, 0.43601068812349186],
    [0.3863350873482144, 0.3248421170772558],
    [0.398537505557738, 0.24444808964901]
])
paper_activation_time = paper_activation[:,0]
paper_activation_vals = paper_activation[:,1]

paper_angularvel = np.array([
    [0.000001047539983291268, -2.542772366942785],
    [0.01349126744481429, 55.09863898367672],
    [0.026922825110580945, 88.16371478628876],
    [0.04018677637902092, 53.43200287025955],
    [0.05346958336716015, 26.32742957257753],
    [0.06720073746814823, -19.420736577739234],
    [0.08048773461622066, -44.83039018245526],
    [0.09379463502397559, -62.189175245582646],
    [0.10715915013081156, -56.2428145304271],
    [0.12056975699691239, -31.652337192645206],
    [0.13397198354314688, -10.451699240795364],
    [0.14789693254104394, 22.190170408566786],
    [0.16080576775514802, 43.814014513666464],
    [0.1737051751094025, 61.624289309592584],
    [0.18659305952384073, 74.77353494986212],
    [0.19945894559862984, 79.02445220205999],
    [0.2133168520375963, 84.54760676396567],
    [0.22667298682456585, 87.10412809318917],
    [0.2410200944357295, 90.50915680887897],
    [0.2538901706704517, 96.45499375404285],
    [0.2672484005373881, 99.85897492974937],
    [0.27912017116803334, 101.99019502575639],
    [0.2934578509193473, 101.58165443227261],
    [0.30976490583924976, 97.78536953282337],
    [0.31963901772175773, 91.86362600727517],
    [0.3340636432916849, 126.62466904283653],
    [0.3468960080870087, 117.31622875130616],
    [0.36062820972798004, 71.9917925242309],
    [0.3744662129072639, 69.46407854454795],
    [0.38726086626318923, 44.90136101632331],
    [0.4004975814920636, -0.8473289039851011]
])
paper_angularvel_time = paper_angularvel[:,0]
paper_angularvel_vals = paper_angularvel[:,1]

test_excitation = np.array([
        [0.0005706532273696552, 0.9558441558441557],
        [0.006540802481101005, 0.9558441558441557],
        [0.010716805582477215, 0.9506493506493506],
        [0.014923822446210502, 0.0025974025974027093],
        [0.020296956774568706, 0.0025974025974027093],
        [0.03462686567164179, 0],
        [0.05194339988369838, 0.0051948051948051965],
        [0.06329908897073074, 0.025974025974025983],
        [0.07585190928474507, 0.051948051948051965],
        [0.08781236673773986, 0.08571428571428563],
        [0.10812327970536924, 0.10649350649350642],
        [0.13380112424888546, 0.11688311688311681],
        [0.15649389416553594, 0.1272727272727272],
        [0.18157317309556115, 0.1350649350649349],
        [0.20665090133746855, 0.14025974025974008],
        [0.22636790075596047, 0.16623376623376607],
        [0.2413196355882924, 0.21038961038961035],
        [0.2687993797247529, 0.23896103896103893],
        [0.3004675324675325, 0.283116883116883],
        [0.31956115526264783, 0.2649350649350649],
        [0.3233665439038574, 0.638961038961039],
        [0.33232176778445444, 0.638961038961039],
        [0.3344632680752084, 0.22597402597402594],
        [0.3361194029850747, 0],
        [0.3438790463268076, 0.0025974025974027093],
        [0.34817910447761197, 0.19999999999999996],
        [0.35653731343283585, 0.19999999999999996],
        [0.3639154874975771, 0.5584415584415583],
        [0.37227524714091886, 0.561038961038961],
        [0.3761194029850747, 0],
        [0.38447761194029856, 0],
        [0.39106028300058154, 0.025974025974025983],
        [0.40002015894553217, 0.03376623376623389]
    ])
test_exc_time = test_excitation[:,0]
test_exc_vals = test_excitation[:,1]

# Calculate MSE
time = sim_test.t
activation_MSE = []
angle_MSE = []
angular_velocity_MSE = []
for i in range(len(time)):
    ci_act = 0
    act_dist = np.absolute(time[i] - paper_activation_time[0])
    ci_angle = 0
    angle_dist = np.absolute(time[i] - paper_angle_time[0])
    ci_angular_vel = 0
    angular_vel_dist = np.absolute(time[i] - paper_angularvel_time[0])
    for j in range(len(paper_activation_time)):
        act_dist_temp = np.absolute(time[i] - paper_activation_time[j])
        if act_dist_temp < act_dist:
            act_dist = act_dist_temp
            ci_act = j
    for j in range(len(paper_angle_time)):
        angle_dist_temp = np.absolute(time[i] - paper_angle_time[j])
        if angle_dist_temp < angle_dist:
            angle_dist = angle_dist_temp
            ci_angle = j
    for j in range(len(paper_angularvel_time)):
        angular_vel_dist_temp = np.absolute(time[i]-paper_angularvel_time[j])
        if angular_vel_dist_temp < angular_vel_dist:
            angular_vel_dist = angular_vel_dist_temp
            ci_angular_vel = j
    activation_MSE.append(paper_activation_vals[ci_act])
    angle_MSE.append(paper_angle_vals[ci_angle])
    angular_velocity_MSE.append(paper_angularvel_vals[ci_angular_vel])

MSE_activation = mean_squared_error(sim_test.y[0], activation_MSE, squared=False)/(np.max(paper_activation_vals)-np.min(paper_activation_vals))
MSE_angle = mean_squared_error(sim_test.y[1], angle_MSE, squared=False)/(np.max(paper_angle_vals)-np.min(paper_angle_vals))
MSE_angular_velocity = mean_squared_error(sim_test.y[2], angular_velocity_MSE, squared=False)/(np.max(paper_angularvel_vals)-np.min(paper_angularvel_vals))

print("NRMSE Activation = " + str(MSE_activation))
print("NRMSE Angle = " + str(MSE_angle))
print("NRMSE Angular Velocity = " + str(MSE_angular_velocity))

plt.subplot(2,2,1)
plt.plot(t_exc, exc_test, drawstyle="steps")
plt.plot(test_exc_time, test_exc_vals, drawstyle="steps")
plt.ylabel('Excitation Level')
plt.xlabel('Time (s)')
plt.subplot(2,2,3)
plt.plot(sim_test.t, sim_test.y[0])
plt.plot(paper_activation_time, paper_activation_vals)
plt.annotate("NRMSE: "+str(np.around(MSE_activation, 3)), xy=(0.23,0))
plt.ylabel('Activation Level')
plt.xlabel('Time (s)')
plt.subplot(2,2,2)
plt.plot(sim_test.t, sim_test.y[1])
plt.plot(paper_angle_time, paper_angle_vals)
plt.annotate("NRMSE: "+str(np.around(MSE_angle, 3)), xy=(0.23,-15.5))
plt.ylabel('Angle (deg)')
plt.xlabel('Time (s)')
plt.subplot(2,2,4)
plt.plot(sim_test.t, sim_test.y[2])
plt.plot(paper_angularvel_time, paper_angularvel_vals)
plt.annotate("NRMSE: "+str(np.around(MSE_angular_velocity, 3)), xy=(0.22,-62))
plt.ylabel('Angular Velocity (deg/s)')
plt.xlabel('Time (s)')
plt.legend(['Simulated', 'Literature'], bbox_to_anchor=(0, -0.5, 1, 0.102), loc='lower left', borderaxespad=0., mode="expand", ncol=2)
plt.tight_layout(h_pad=0.4)
plt.show()

# Truncate the plots

MSE_act_trunc = mean_squared_error(sim_test.y[0][13:], activation_MSE[13:], squared=False)/(np.max(paper_activation_vals)-np.min(paper_activation_vals))
MSE_angle_trunc = mean_squared_error(sim_test.y[1][13:], angle_MSE[13:], squared=False)/(np.max(paper_angle_vals)-np.min(paper_angle_vals))
MSE_angular_velocity_trunc = mean_squared_error(sim_test.y[2][13:], angular_velocity_MSE[13:], squared=False)/(np.max(paper_angularvel_vals)-np.min(paper_angularvel_vals))

print("Truncated NRMSE Activation = " + str(MSE_act_trunc))
print("Truncated NRMSE Angle = " + str(MSE_angle_trunc))
print("Truncated NRMSE Angular Velocity = " + str(MSE_angular_velocity_trunc))

plt.subplot(2,2,1)
plt.plot(t_exc[10:], exc_test[10:], drawstyle="steps")
plt.plot(test_exc_time[10:], test_exc_vals[10:], drawstyle="steps")
plt.ylabel('Excitation Level')
plt.xlabel('Time (s)')
plt.subplot(2,2,3)
plt.plot(sim_test.t[13:], sim_test.y[0][13:])
plt.plot(paper_activation_time[8:], paper_activation_vals[8:])
plt.ylim(0, 0.7)
plt.annotate("NRMSE: "+str(np.around(MSE_act_trunc, 3)), xy=(0.28,0.01))
plt.ylabel('Activation Level')
plt.xlabel('Time (s)')
plt.subplot(2,2,2)
plt.plot(sim_test.t[13:], sim_test.y[1][13:])
plt.plot(paper_angle_time[8:], paper_angle_vals[8:])
plt.annotate("NRMSE: "+str(np.around(MSE_angle_trunc, 3)), xy=(0.28,-15))
plt.ylabel('Angle (deg)')
plt.xlabel('Time (s)')
plt.subplot(2,2,4)
plt.plot(sim_test.t[13:], sim_test.y[2][13:])
plt.plot(paper_angularvel_time[8:], paper_angularvel_vals[8:])
plt.annotate("NRMSE: "+str(np.around(MSE_angular_velocity_trunc, 3)), xy=(0.28,-60))
plt.ylabel('Angular Velocity (deg/s)')
plt.xlabel('Time (s)')
plt.legend(['Simulated', 'Literature'], bbox_to_anchor=(0, -0.5, 1, 0.102), loc='lower left', borderaxespad=0., mode="expand", ncol=2)
plt.tight_layout(h_pad=0.4)
plt.show()
