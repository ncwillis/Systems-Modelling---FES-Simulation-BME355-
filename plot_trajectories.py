import matplotlib.pyplot as plt
import simulation as sim
import numpy as np
import sim_regressions as reg

def process_pulse_trains(traj):
    """
    :param traj: Amplitude data for pulse train
    :return: Amplitude data with prepended and appended zero (for plotting)
    """
    if isinstance(traj, list):
        data = traj
    else:
        data = np.ndarray.tolist(traj)
    data.insert(0,0)
    data.append(0)
    return data

def plot_FES_pulsetrains():
    """
    void function to plot different pulse train shapes
    """
    plt.subplot(2,2,1)
    plt.plot(pulse_times, pt_unique, drawstyle="steps", color='red')
    plt.fill_between(pulse_times, pt_unique, step="pre", alpha=0.4, color='red')
    plt.ylim(0, 350)
    plt.ylabel('Pulse Width (microseconds)')
    plt.xlabel('Time (s)')
    plt.title('Uniquely Generated Pulse Train')
    plt.subplot(2,2,2)
    plt.plot(pulse_times, pt_rectangular, drawstyle="steps", color='red')
    plt.fill_between(pulse_times, pt_rectangular, step="pre", alpha=0.4, color='red')
    plt.ylim(0, 350)
    plt.ylabel('Pulse Width (microseconds)')
    plt.xlabel('Time (s)')
    plt.title('Rectangular Pulse Train')

    plt.subplot(2,2,3)
    plt.plot(pulse_times, pt_triangular, drawstyle="steps", color='red')
    plt.fill_between(pulse_times, pt_triangular, step="pre", alpha=0.4, color='red')
    plt.ylim(0, 350)
    plt.ylabel('Pulse Width (microseconds)')
    plt.xlabel('Time (s)')
    plt.title('Triangular Pulse Train')

    plt.subplot(2,2,4)
    plt.plot(pulse_times, pt_trapezoidal, drawstyle="steps", color='red')
    plt.fill_between(pulse_times, pt_trapezoidal, step="pre", alpha=0.4, color='red')
    plt.ylim(0, 350)
    plt.xlabel('Time (s)')
    plt.ylabel('Pulse Width (microseconds)')
    plt.title('Trapezoidal Pulse Train')
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=0)
    plt.show()

def get_excitation_results():
    """
    :return: excitation trajectory data for each signal shape
    """
    t_exc = np.linspace(0,0.4,40)
    exc_tri = np.zeros(len(t_exc))
    exc_rec = np.zeros(len(t_exc))
    exc_uniq = np.zeros(len(t_exc))
    exc_trap = np.zeros(len(t_exc))
    for i in range(len(t_exc)):
        exc_tri[i] = sim.excitation_triangular.eval(t_exc[i])
        exc_rec[i] = sim.excitation_rectangular.eval(t_exc[i])
        exc_uniq[i] = sim.excitation_unique.eval(t_exc[i])
        exc_trap[i] = sim.excitation_trapezoidal.eval(t_exc[i])
    return t_exc, exc_tri, exc_rec, exc_uniq, exc_trap

def plot_excitation_trajectories():
    """
    void function to plot excitation trajectories
    """
    plt.plot(t_exc, exc_uniq)
    plt.plot(t_exc, exc_rec)
    plt.plot(t_exc, exc_tri)
    plt.plot(t_exc, exc_trap)
    plt.title('Excitation')
    plt.ylabel('Normalized Excitation')
    plt.xlabel('Time (s)')
    plt.legend(['Unique', 'Rectangular', 'Triangular', 'Trapezoidal'])
    plt.show()

def get_sim_results():
    """
    :return: simulation results for each signal type
    """
    sim_unique = sim.sim_unique
    sim_rectangular = sim.sim_rectangular
    sim_triangular = sim.sim_triangular
    sim_trapezoidal = sim.sim_trapezoidal
    # sim_zero = sim.sim_zero
    # sim_test = sim.sim_test
    return sim_unique, sim_rectangular, sim_triangular, sim_trapezoidal

def get_natural_trajectories():
    """
    :return: Expected/natural gait trajectories
    """
    t_expected = np.linspace(0, 0.4, 40)
    exp_angle = reg.get_angle_regression()
    exp_angular_velocity = reg.get_angular_velocity_regression()
    exp_angle_traj = []
    exp_angular_vel_traj = []
    for i in range(len(t_expected)):
        exp_angle_traj.append(exp_angle.eval(t_expected[i]))
        exp_angular_vel_traj.append(exp_angular_velocity.eval(t_expected[i]))
    return t_expected, exp_angle_traj, exp_angular_vel_traj

def plot_activation():
    """
    void function to plot activation trajectories
    """
    plt.plot(sim_unique.t, sim_unique.y[0])
    plt.plot(sim_rectangular.t, sim_rectangular.y[0])
    plt.plot(sim_triangular.t, sim_triangular.y[0])
    plt.plot(sim_trapezoidal.t, sim_trapezoidal.y[0])
    plt.title('Activation')
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Activation')
    plt.legend(['Unique', 'Rectangular', 'Triangular', 'Trapezoidal'])
    plt.show()

def plot_ankle_posn_velocity():
    """
    void function to plot ankle angle and angular velocity trajectories
    """
    plt.subplot(1,2,1)
    plt.plot(sim_unique.t, sim_unique.y[1])
    plt.plot(sim_rectangular.t, sim_rectangular.y[1])
    plt.plot(sim_triangular.t, sim_triangular.y[1])
    plt.plot(sim_trapezoidal.t, sim_trapezoidal.y[1])
    plt.plot(t_expected, exp_angle_traj)
    plt.legend(['Unique', 'Rectangular', 'Triangular', 'Trapezoidal', 'Natural'])
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (deg)')
    plt.subplot(1,2,2)
    plt.plot(sim_unique.t, sim_unique.y[2])
    plt.plot(sim_rectangular.t, sim_rectangular.y[2])
    plt.plot(sim_triangular.t, sim_triangular.y[2])
    plt.plot(sim_trapezoidal.t, sim_trapezoidal.y[2])
    plt.plot(t_expected, exp_angular_vel_traj)
    plt.legend(['Unique', 'Rectangular', 'Triangular', 'Trapezoidal', 'Natural'])
    plt.xlabel('Time (s)')
    plt.ylabel('Angular Velocity (deg/s)')
    plt.show()

if __name__ == "__main__":
    pulse_times = process_pulse_trains(sim.pulse_times)
    pt_unique = process_pulse_trains(sim.pulse_train_unique)
    pt_rectangular = process_pulse_trains(sim.pulse_train_rectangular)
    pt_triangular = process_pulse_trains(sim.pulse_train_rectangular)
    pt_trapezoidal = process_pulse_trains(sim.pulse_train_trapezoidal)

    plot_FES_pulsetrains()

    t_exc, exc_tri, exc_rec, exc_uniq, exc_trap = get_excitation_results()
    plot_excitation_trajectories()

    sim_unique, sim_rectangular, sim_triangular, sim_trapezoidal = get_sim_results()
    t_expected, exp_angle_traj, exp_angular_vel_traj = get_natural_trajectories()

    plot_activation()
    plot_ankle_posn_velocity()