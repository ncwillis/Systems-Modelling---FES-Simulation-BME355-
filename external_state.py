import numpy as np
import sim_regressions as reg
import csv

def get_x_ext_1():
    """
    :return: Regression objection for external state vector 1
    """
    x_ext1_trajectory = []
    with open('x_ext_1.txt') as csv_file:
        csv_read = csv.reader(csv_file, delimiter=',')
        for row in csv_read:
            row_content = [float(row[0]), float(row[1])]
            x_ext1_trajectory.append(row_content)
    x_ext1_trajectory = np.array(x_ext1_trajectory)

    time = x_ext1_trajectory[:,0]
    x_ext_1 = x_ext1_trajectory[:,1]

    centres = np.arange(0, 0.4, .05)
    width = .05
    result = reg.Regression(time, x_ext_1, centres, width, .075, sigmoids=False)

    return result

def get_x_ext_2():
    """
    :return: Regression objection for external state vector 2
    """
    x_ext2_trajectory = []
    with open('x_ext_2.txt') as csv_file:
        csv_read = csv.reader(csv_file, delimiter=',')
        for row in csv_read:
            row_content = [float(row[0]), float(row[1])]
            x_ext2_trajectory.append(row_content)
    x_ext2_trajectory = np.array(x_ext2_trajectory)

    time = x_ext2_trajectory[:,0]
    x_ext_2 = x_ext2_trajectory[:,1]

    centres = np.arange(0, 0.4, .05)
    width = .05
    result = reg.Regression(time, x_ext_2, centres, width, .075, sigmoids=False)

    return result

def get_x_ext_3():
    """
    :return: Regression objection for external state vector 3
    """
    x_ext3_trajectory = []
    with open('x_ext_3.txt') as csv_file:
        csv_read = csv.reader(csv_file, delimiter=',')
        for row in csv_read:
            row_content = [float(row[0]), float(row[1])]
            x_ext3_trajectory.append(row_content)
    x_ext3_trajectory = np.array(x_ext3_trajectory)

    time = x_ext3_trajectory[:,0]
    x_ext_3 = x_ext3_trajectory[:,1]

    centres = np.arange(0, 0.4, .05)
    width = .05
    result = reg.Regression(time, x_ext_3, centres, width, .075, sigmoids=False)

    return result

def get_x_ext_4(time, x_ext_3_traj):
    """
    :return: Regression objection for external state vector 4
    """
    x_ext_3_traj = np.array(x_ext_3_traj)
    dy = np.zeros(x_ext_3_traj.shape, np.float)
    dy[0:-1] = np.diff(x_ext_3_traj)/np.diff(time)
    dy[-1] = (x_ext_3_traj[-1] - x_ext_3_traj[-2])/(time[-1] - time[-2])
    dy = np.array(dy)

    x_ext_4 = dy

    centres = np.arange(0, 0.4, .05)
    width = .05
    result = reg.Regression(time, x_ext_4, centres, width, .075, sigmoids=False)

    return result

x_ext_1 = get_x_ext_1()
x_ext_2 = get_x_ext_2()
x_ext_3 = get_x_ext_3()
time = np.arange(0, 0.4, 0.01)
x_ext_1_traj = []
x_ext_2_traj = []
x_ext_3_traj = []

for i in range(len(time)):
    x_ext_1_traj.append(x_ext_1.eval(time[i])[0])
    x_ext_2_traj.append(x_ext_2.eval(time[i])[0])
    x_ext_3_traj.append(x_ext_3.eval(time[i])[0])

x_ext_4 = get_x_ext_4(time, x_ext_3_traj)
x_ext_4_traj = []

for i in range(len(time)):
    x_ext_4_traj.append(x_ext_4.eval(time[i])[0])
