import numpy as np
import matplotlib.pyplot as plt
import simulation as sim
import sim_regressions as reg
from shapely.geometry import Polygon

# Get activation time integrals
t_unique = sim.sim_unique.t
act_unique = sim.sim_unique.y[0]

t_rect = sim.sim_rectangular.t
act_rect = sim.sim_rectangular.y[0]

t_triang = sim.sim_triangular.t
act_triang = sim.sim_triangular.y[0]

t_trap = sim.sim_trapezoidal.t
act_trap = sim.sim_trapezoidal.y[0]

ati_unique = np.trapz(act_unique, t_unique)
ati_rect = np.trapz(act_rect, t_rect)
ati_triang = np.trapz(act_triang, t_triang)
ati_trap = np.trapz(act_trap, t_trap)

# Plot activation results
plt.plot(t_unique, act_unique)
plt.plot(t_rect, act_rect)
plt.plot(t_triang, act_triang)
plt.plot(t_trap, act_trap)
plt.title('Activation')
plt.xlabel('Time (s)')
plt.ylabel('Normalized Activation')
plt.legend(['Unique', 'Rectangular', 'Triangular', 'Trapezoidal'])
plt.show()

print("ATI Unique = " + str(ati_unique))
print("AIT Rectangular = " + str(ati_rect))
print("ATI Triangular = " + str(ati_triang))
print("ATI Trapezoidal = " + str(ati_trap))

# Plot ATI results
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

#Get area between expected and actual results (angle)
angle = reg.get_angle_regression()
angle_natural = []
polygon_unique_pts = []
angle_unique = sim.sim_unique.y[1]
for i in range(len(t_unique)):
    angle_natural.append(angle.eval(t_unique[i]))
    polygon_unique_pts.append([t_unique[i], angle_unique[i]])
    polygon_unique_pts.append([t_unique[i], angle_natural[i][0]])

polygon_unique = Polygon(polygon_unique_pts)
area_unique = polygon_unique.area

polygon_rect_pts = []
angle_natural = []
angle_rect = sim.sim_rectangular.y[1]
for i in range(len(t_rect)):
    angle_natural.append(angle.eval(t_rect[i]))
    polygon_rect_pts.append([t_rect[i], angle_rect[i]])
    polygon_rect_pts.append([t_rect[i], angle_natural[i][0]])

polygon_rect = Polygon(polygon_rect_pts)
area_rect = polygon_rect.area

polygon_triang_pts = []
angle_natural = []
angle_triang = sim.sim_triangular.y[1]
for i in range(len(t_triang)):
    angle_natural.append(angle.eval(t_triang[i]))
    polygon_triang_pts.append([t_triang[i], angle_triang[i]])
    polygon_triang_pts.append([t_triang[i], angle_natural[i][0]])

polygon_triang = Polygon(polygon_triang_pts)
area_triang = polygon_triang.area

polygon_trap_pts = []
angle_natural = []
angle_trap = sim.sim_trapezoidal.y[1]
for i in range(len(t_trap)):
    angle_natural.append(angle.eval(t_trap[i]))
    polygon_trap_pts.append([t_trap[i], angle_trap[i]])
    polygon_trap_pts.append([t_trap[i], angle_natural[i][0]])

polygon_trap = Polygon(polygon_trap_pts)
area_trap = polygon_trap.area

print(area_unique)
print(area_rect)
print(area_triang)
print(area_trap)

# polygon_unique_pts = np.array(polygon_unique_pts)
# plt.plot(polygon_unique_pts[:,0], polygon_unique_pts[:,1])
# plt.show()