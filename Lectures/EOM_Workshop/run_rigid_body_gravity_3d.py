import rigid_body as rb
import numpy as np
import integrators as intg
import aerosonde_uav as uav
import matplotlib.pyplot as plt

yaw0 = np.deg2rad(0)
pitch0 = np.deg2rad(0)
roll0 = np.deg2rad(0)
quat0 = rb.quat_from_ypr(yaw0,pitch0,roll0)

#x0 = np.array([0.0,0.0,0.0, 1.0,0.0,0.0, 1.0,0.0,0.0,0.0, 0.0,0.0,0.0])

x0 = np.array([0.0,0.0,0.0,\
               10,10,10.0,\
               float(quat0[0]),float(quat0[1]),float(quat0[2]),float(quat0[3]),\
               np.deg2rad(0),np.deg2rad(0),np.deg2rad(0)])
x0 = x0.reshape(x0.shape[0],1)

drone = rb.RigidBody(uav.m,uav.J,x0)

t = 0
u = np.array([0.0,0.0,-drone.m*32.174, 0.0,0.0,0.0])
u = u.reshape(u.shape[0],1)

tf = 2*x0[5]/32.174; dt = 1e-3; n = int(np.floor(tf/dt));
integrator = intg.RungeKutta4(dt, drone.f)

x = x0
t_history = [0]
x_history = [x]

for i in range(n):
    x = integrator.step(t, drone.x, u)
    t = (i+1) * dt
    t_history.append(t)
    x_history.append(x)
    
    #Update drone state
    drone.x = x
    #print(drone.x)


print(x)
print(rb.get_euler_angles_from_rot(rb.rot_from_quat(x[6:10])))
print(all(x == drone.x))

t_ = np.asarray(t_history)
x_ = np.asarray(x_history)

# close all figures
plt.close('all')

plt.figure()
plt.plot(t_history, x_[:,3],label = 'u')
plt.plot(t_history, x_[:,4],label = 'v')
plt.plot(t_history, x_[:,5],label = 'w')
plt.xlabel('t')
plt.ylabel('v')
plt.legend()
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_[:,0], x_[:,1],x_[:,2],'b',label = 'traj')
plt.xlabel('d_x')
plt.ylabel('d_y')
plt.legend()
ax.show()