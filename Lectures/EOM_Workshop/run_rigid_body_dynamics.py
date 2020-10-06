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
               np.deg2rad(0),np.deg2rad(0),np.deg2rad(0)])        
x0 = x0.reshape(x0.shape[0],1)

drone = rb.RigidBody(uav.m,uav.J,x0)
t = 0

# u = [fx,fy,fz,Mx,My,Mz]
"""
u = np.array([uav.m,0,0,\
              0,0,0]) # for linear dynamics
u = u.reshape(u.shape[0],1)
"""
"""
u = np.array([0,0,0,\
              uav.J[0,0],0,0]) # for rotational dynamics
u = u.reshape(u.shape[0],1)
"""

tf = 10; dt = 1e-2; n = int(np.floor(tf/dt));

# white noise input
np.random.seed(1)
white_noise = np.random.normal(0, 1, n)
u = np.zeros([6,n])
u[0] = white_noise

integrator = intg.RungeKutta4(dt, drone.f)

x = x0
t_history = [0]
x_history = [x]

for i in range(n):
    """
    #Only use this for non varying input force/moment
    x = integrator.step(t, drone.x, u)
    """

    #Only use this for any time varying input force/moment
    a = u[:,i].reshape(u[:,i].shape[0],1)
    x = integrator.step(t, drone.x, a)
    
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
plt.plot(t_history,x_history[:,0],label="x")
plt.plot(t_history,x_history[:,1],label="y")
plt.plot(t_history,x_history[:,2],label="z",linestyle="--")
plt.legend()
plt.title("Position vs Time")
#plt.savefig("pos_vs_time_fx_m") #n=1000, dt = 0.01
plt.savefig("pos_vs_time_white_noise") #n=1000, dt = 0.01

plt.figure()
plt.plot(t_history,x_history[:,3],label="u")
plt.plot(t_history,x_history[:,4],label="v")
plt.plot(t_history,x_history[:,5],label="w",linestyle="--")
plt.legend()
plt.title("Velocity vs Time")
#plt.savefig("vel_vs_time_fx_m") #n=1000, dt = 0.01
plt.savefig("vel_vs_time_white_noise") #n=1000, dt = 0.01

plt.figure()
plt.plot(t_history,angles[:,0],label="roll")
plt.plot(t_history,angles[:,1],label="pitch")
plt.plot(t_history,angles[:,2],label="yaw",linestyle="--")
plt.legend()
plt.title("Euler Angles vs Time")

plt.figure()
plt.plot(t_history,x_history[:,10],label="p")
plt.plot(t_history,x_history[:,11],label="q")
plt.plot(t_history,x_history[:,12],label="r",linestyle="--")
plt.legend()
plt.title("Angular Velocity vs Time")

plt.show()


