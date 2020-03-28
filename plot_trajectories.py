import matplotlib.pyplot as plt
import simulation as sim
import numpy as np

# Get FES pulse trains
pulse_times = np.ndarray.tolist(sim.pulse_times)
pulse_times.insert(0, 0)
pulse_times.append(0)
pt_unique = np.ndarray.tolist(sim.pulse_train_unique)
pt_unique.insert(0,0)
pt_unique.append(0)
pt_rectangular = np.ndarray.tolist(sim.pulse_train_rectangular)
pt_rectangular.insert(0,0)
pt_rectangular.append(0)
pt_triangular = sim.pulse_train_triangular
pt_triangular.insert(0,0)
pt_triangular.append(0)
pt_trapezoidal = sim.pulse_train_trapezoidal
pt_trapezoidal.insert(0,0)
pt_trapezoidal.append(0)


# Plot FES Signal Pulse Trains
plt.subplot(4,1,1)
plt.plot(pulse_times, pt_unique, drawstyle="steps", color='red')
plt.fill_between(pulse_times, pt_unique, step="pre", alpha=0.4, color='red')
plt.ylim(0, 350)
# plt.ylabel('Pulse Width (microseconds)')
plt.title('Uniquely Generated Pulse Train')

plt.subplot(4,1,2)
plt.plot(pulse_times, pt_rectangular, drawstyle="steps", color='red')
plt.fill_between(pulse_times, pt_rectangular, step="pre", alpha=0.4, color='red')
plt.ylim(0, 350)
# plt.ylabel('Pulse Width (microseconds)')
plt.title('Rectangular Pulse Train')

plt.subplot(4,1,3)
plt.plot(pulse_times, pt_triangular, drawstyle="steps", color='red')
plt.fill_between(pulse_times, pt_triangular, step="pre", alpha=0.4, color='red')
plt.ylim(0, 350)
# plt.ylabel('Pulse Width (microseconds)')
plt.title('Triangular Pulse Train')

plt.subplot(4,1,4)
plt.plot(pulse_times, pt_trapezoidal, drawstyle="steps", color='red')
plt.fill_between(pulse_times, pt_trapezoidal, step="pre", alpha=0.4, color='red')
plt.ylim(0, 350)
plt.xlabel('Time (s)')
plt.ylabel('Pulse Width (microseconds)')
plt.title('Trapezoidal Pulse Train')
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=0)
plt.show()

# Get excitation trajectories
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

# Plot excitation trajectories
plt.plot(t_exc, exc_uniq)
plt.plot(t_exc, exc_rec)
plt.plot(t_exc, exc_tri)
plt.plot(t_exc, exc_trap)
plt.title('Excitation')
plt.xlabel('Normalized Excitation')
plt.ylabel('Time (s)')
plt.legend(['Unique', 'Rectangular', 'Triangular', 'Trapezoidal'])
plt.show()

# Get simulation results
sim_unique = sim.sim_unique
sim_rectangular = sim.sim_rectangular
sim_triangular = sim.sim_triangular
sim_trapezoidal = sim.sim_trapezoidal

# Plot activation results
plt.plot(sim_unique.t, sim_unique.y[0])
plt.plot(sim_rectangular.t, sim_rectangular.y[0])
plt.plot(sim_triangular.t, sim_triangular.y[0])
plt.plot(sim_trapezoidal.t, sim_trapezoidal.y[0])
plt.title('Activation')
plt.xlabel('Time (s)')
plt.ylabel('Normalized Activation')
plt.legend(['Unique', 'Rectangular', 'Triangular', 'Trapezoidal'])
plt.show()

# Plot ankle angle and angular velocity results
plt.subplot(2,1,1)
plt.plot(sim_unique.t, sim_unique.y[1])
plt.plot(sim_rectangular.t, sim_rectangular.y[1])
plt.plot(sim_triangular.t, sim_triangular.y[1])
plt.plot(sim_trapezoidal.t, sim_trapezoidal.y[1])
plt.legend(['Unique', 'Rectangular', 'Triangular', 'Trapezoidal'])
plt.ylabel('Angle (degrees)')
plt.subplot(2,1,2)
plt.plot(sim_unique.t, sim_unique.y[2])
plt.plot(sim_rectangular.t, sim_rectangular.y[2])
plt.plot(sim_triangular.t, sim_triangular.y[2])
plt.plot(sim_trapezoidal.t, sim_trapezoidal.y[2])
plt.legend(['Unique', 'Rectangular', 'Triangular', 'Trapezoidal'])
plt.xlabel('Time (s)')
plt.ylabel('Anglular Velocity (degree/s)')
plt.show()