import rigid_body as rb
import numpy as np
import integrators as intg
import aerosonde_uav as uav
import matplotlib.pyplot as plt

# Initial condition of angles
yaw0 = np.deg2rad(0)
pitch0 = np.deg2rad(0)
roll0 = np.deg2rad(0)
quat0 = rb.quat_from_ypr(yaw0,pitch0,roll0)

# X0 = [x,y,z,
#       u,v,w,
#       quat1,quat2,quat3, quat4,
#       p,q,r]

x0 = np.array([0,0,0,\
               0,0,0,\
               float(quat0[0]),float(quat0[1]),float(quat0[2]),float(quat0[3]),\
               np.deg2rad(10),np.deg2rad(0),np.deg2rad(0)])        
x0 = x0.reshape(x0.shape[0],1)

drone = rb.RigidBody(uav.m,uav.J,x0)
t = 0
# u = [fx,fy,fz,Mx,My,Mz]
u = np.array([0,0,0,\
              0,0,0])
u = u.reshape(u.shape[0],1)

tf = 10; dt = 1e-2; n = int(np.floor(tf/dt));
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


t_history = np.asarray(t_history)
x_history = np.asarray(x_history)
# Extracting angles from the quaternion components
angles = np.empty([x_history.shape[0],3])
for i in range(x_history.shape[0]):
    angles[i,:] =  np.rad2deg(rb.get_euler_angles_from_rot(rb.rot_from_quat(x_history[i,6:10])))

# Plot state vs time
plt.figure()
plt.scatter(t_history,x_history[:,0],label="x",s=0.1)
plt.scatter(t_history,x_history[:,1],label="y",s=0.1)
plt.scatter(t_history,x_history[:,2],label="z",s=0.1)
plt.legend()
plt.title("Position vs time plot")

plt.figure()
plt.scatter(t_history,x_history[:,3],label="u",s=0.1)
plt.scatter(t_history,x_history[:,4],label="v",s=0.1)
plt.scatter(t_history,x_history[:,5],label="w",s=0.1)
plt.legend()
plt.title("Velocity vs time plot")

plt.figure()
plt.scatter(t_history,angles[:,0],label="roll",s=0.1)
plt.scatter(t_history,angles[:,1],label="pitch",s=0.1)
plt.scatter(t_history,angles[:,2],label="yaw",s=0.1)
plt.legend()
plt.title("Euler angle vs time plot")

plt.figure()
plt.scatter(t_history,x_history[:,10],label="p",s=0.1)
plt.scatter(t_history,x_history[:,11],label="q",s=0.1)
plt.scatter(t_history,x_history[:,12],label="r",s=0.1)
plt.legend()
plt.title("Angular velocity vs time plot")

plt.show()