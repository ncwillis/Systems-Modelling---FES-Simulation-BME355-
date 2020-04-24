import matplotlib.pyplot as plt
import numpy as np
import simulation as sim
from sklearn.metrics import mean_squared_error
import csv

def parse_lit_data(file_name):
    data = []
    with open(file_name) as csv_file:
        csv_read = csv.reader(csv_file, delimiter=',')
        for row in csv_read:
            row_content = [float(row[0]), float(row[1])]
            data.append(row_content)
    data = np.array(data)
    data_time = data[:, 0]
    data_vals = data[:, 1]
    return data_time, data_vals

def calculate_MSE(data_time, data_vals, sim_index, trunc):
    time = sim_test.t
    MSE = []
    for i in range(len(time)):
        ci = 0
        dist = np.absolute(time[i] - data_time[0])
        for j in range(len(data_time)):
            dist_temp = np.absolute(time[i] - data_time[j])
            if dist_temp < dist:
                dist = dist_temp
                ci = j
        MSE.append(data_vals[ci])
    if trunc is False:
        MSE_val = mean_squared_error(sim_test.y[sim_index], MSE, squared=False) / (
                    np.max(data_vals) - np.min(data_vals))
    else:
        MSE_val = mean_squared_error(sim_test.y[sim_index][13:], MSE[13:], squared=False) / (
                    np.max(data_vals) - np.min(data_vals))
    return MSE_val

def plot_validation():
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

    print("NRMSE Activation = " + str(MSE_activation))
    print("NRMSE Angle = " + str(MSE_angle))
    print("NRMSE Angular Velocity = " + str(MSE_angular_velocity))

def plot_validation_truncated():
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

    print("Truncated NRMSE Activation = " + str(MSE_act_trunc))
    print("Truncated NRMSE Angle = " + str(MSE_angle_trunc))
    print("Truncated NRMSE Angular Velocity = " + str(MSE_angular_velocity_trunc))

t_exc = np.linspace(0,0.4,40)

exc_test = np.zeros(len(t_exc))
for i in range(len(t_exc)):
    exc_test[i] = sim.excitation_test.eval(t_exc[i])

sim_test = sim.sim_test

paper_angle_time, paper_angle_vals = parse_lit_data('lit_ankle_angle.txt')
paper_activation_time, paper_activation_vals = parse_lit_data('lit_activation.txt')
paper_angularvel_time, paper_angularvel_vals = parse_lit_data('lit_ankle_angularvelocity.txt')
test_exc_time, test_exc_vals = parse_lit_data('lit_excitation.txt')

MSE_activation = calculate_MSE(paper_activation_time, paper_activation_vals, 0, False)
MSE_angle = calculate_MSE(paper_angle_time, paper_angle_vals, 1, False)
MSE_angular_velocity = calculate_MSE(paper_angularvel_time, paper_angularvel_vals, 2, False)
MSE_act_trunc = calculate_MSE(paper_activation_time, paper_activation_vals, 0, True)
MSE_angle_trunc = calculate_MSE(paper_angle_time, paper_angle_vals, 1, True)
MSE_angular_velocity_trunc = calculate_MSE(paper_angularvel_time, paper_angularvel_vals, 2, True)

plot_validation()
plot_validation_truncated()