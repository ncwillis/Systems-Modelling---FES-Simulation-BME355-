import numpy as np
import matplotlib.pyplot as plt
import simulation as sim
import sim_regressions as reg
from shapely.geometry import Polygon

def get_ATI(simulation):
    """
    :param simulation: Simulation results (array) for given signal shape
    :return: Activation-time integral
    """
    time = simulation.t
    activation = simulation.y[0]
    ati = np.trapz(activation, time)
    return ati

def plot_ati_results():
    """
    void function to plot ATI results
    """
    ati = [ati_unique, ati_rect, ati_triang, ati_trap]
    bars = ('Unique', 'Rectangular', 'Triangular', 'Trapezoidal')
    y_pos = np.arange(len(bars))
    plt.barh(y_pos, ati, color = 'teal', height=0.4)
    plt.text(ati_unique, 0, str(np.round(ati_unique, 2)))
    plt.text(ati_rect, 1, str(np.round(ati_rect, 2)))
    plt.text(ati_triang, 2, str(np.round(ati_triang, 2)))
    plt.text(ati_trap, 3, str(np.round(ati_trap, 2)))
    plt.xlabel('Activation-Time Integral')
    plt.yticks(y_pos, bars)
    plt.show()

def get_area_angles(simulation):
    """
    :param simulation: Simulation results (array) for given signal shape
    :return: Area between simulated and natural ankle angle trajectories
    """
    time = simulation.t
    angles = simulation.y[1]
    angle_exp = reg.get_angle_regression()
    polygon_pts = []
    angle_natural = []
    for i in range(len(time)):
        angle_natural.append(angle_exp.eval(time[i]))
        polygon_pts.append([time[i], angles[i]])
        polygon_pts.append([time[i], angle_natural[i][0]])
    polygon = Polygon(polygon_pts)
    area = polygon.area
    return area

def get_area_angular_velocity(simulation):
    """
    :param simulation: Simulation results (array) for given signal shape
    :return: Area between simulated and natural angular velocity trajectories
    """
    time = simulation.t
    angular_velocity = simulation.y[2]
    angularvel_exp = reg.get_angular_velocity_regression()
    polygon_pts = []
    angularvel_natural = []
    for i in range(len(time)):
        angularvel_natural.append(angularvel_exp.eval(time[i]))
        polygon_pts.append([time[i], angular_velocity[i]])
        polygon_pts.append([time[i], angularvel_natural[i][0]])
    polygon = Polygon(polygon_pts)
    area = polygon.area
    return area

def cost_function(ati, angle_diff, velocity_diff):
    """
    :param ati: Activation time integral for given signal shape
    :param angle_diff: Area between simulated and natural ankle angle trajectories for given signal shape
    :param velocity_diff: Area between simulation and natural angular velocity trajectories for given signal shape
    :return: Result of the cost function
    """
    angle_norm = np.max([area_angle_unique, area_angle_rect, area_angle_triang, area_angle_trap])
    vel_norm = np.max([area_vel_unique, area_vel_rect, area_vel_triang, area_vel_trap])
    act_norm = np.max([ati_unique, ati_rect, ati_triang, ati_trap])
    c1 = 1
    c2 = 0.5
    c3 = 0.25
    return c1 * (ati / act_norm) + c2 * (angle_diff / angle_norm) + c3 * (velocity_diff / vel_norm)

def print_results():
    """
    void function to print activation time integral and cost function results
    """
    print("ATI Unique = " + str(ati_unique))
    print("AIT Rectangular = " + str(ati_rect))
    print("ATI Triangular = " + str(ati_triang))
    print("ATI Trapezoidal = " + str(ati_trap))
    print("Cost Function - Unique: " + str(cost_unique))
    print("Cost Function - Rectangular: " + str(cost_rect))
    print("Cost Function - Triangular: " + str(cost_triang))
    print("Cost Function - Trapezoidal: " + str(cost_trap))

ati_unique = get_ATI(sim.sim_unique)
ati_rect = get_ATI(sim.sim_rectangular)
ati_triang = get_ATI(sim.sim_triangular)
ati_trap = get_ATI(sim.sim_trapezoidal)

area_angle_unique = get_area_angles(sim.sim_unique)
area_angle_rect = get_area_angles(sim.sim_rectangular)
area_angle_triang = get_area_angles(sim.sim_triangular)
area_angle_trap = get_area_angles(sim.sim_trapezoidal)

area_vel_unique = get_area_angular_velocity(sim.sim_unique)
area_vel_rect = get_area_angular_velocity(sim.sim_rectangular)
area_vel_triang = get_area_angular_velocity(sim.sim_triangular)
area_vel_trap = get_area_angular_velocity(sim.sim_trapezoidal)

cost_unique = cost_function(ati_unique, area_angle_unique, area_vel_unique)
cost_rect = cost_function(ati_rect, area_angle_rect, area_vel_rect)
cost_triang = cost_function(ati_triang, area_angle_triang, area_vel_triang)
cost_trap = cost_function(ati_trap, area_angle_trap, area_vel_trap)

print_results()
plot_ati_results()

