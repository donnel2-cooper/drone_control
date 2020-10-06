## Unit tests for rigid_body functions
import rigid_body as rb
import numpy as np
import aerosonde_uav as uav

def within_tol(a,b,tol):
    if (np.abs(a-b)<tol).all():
        return True
    else:
        return False

## Init checks matrix
checks = [True]

## Specify tol on checks
tol = 1e-8

## Unit tests for rb.get_euler_angles_from_rot
checks.append(within_tol(rb.get_euler_angles_from_rot(np.eye(3)),\
                         np.array([0,0,0]),tol))
checks.append(within_tol(rb.get_euler_angles_from_rot(\
                         np.matrix([[0,-1,0],\
                                    [1,0,0],\
                                    [0,0,1]])),np.array([np.pi/2,0,0]),tol))
checks.append(within_tol(rb.get_euler_angles_from_rot(\
                         np.matrix([[-1,0,0],\
                                    [0,-1,0],\
                                    [0,0,1]])),np.array([np.pi,0,0]),tol))
checks.append(within_tol(rb.get_euler_angles_from_rot(\
                         np.matrix([[0,1,0],\
                                    [-1,0,0],\
                                    [0,0,1]])),np.array([-np.pi/2,0,0]),tol))
checks.append(within_tol(rb.get_euler_angles_from_rot(\
                         np.matrix([[0,0,1],\
                                    [0,1,0],\
                                    [-1,0,0]])),np.array([0,np.pi/2,0]),tol))
checks.append(within_tol(rb.get_euler_angles_from_rot(\
                         np.matrix([[0,0,-1],\
                                    [0,1,0],\
                                    [1,0,0]])),np.array([0,-np.pi/2,0]),tol))
checks.append(within_tol(rb.get_euler_angles_from_rot(\
                         np.matrix([[1,0,0],\
                                    [0,0,-1],\
                                    [0,1,0]])),np.array([0,0,np.pi/2]),tol))
checks.append(within_tol(rb.get_euler_angles_from_rot(\
                         np.matrix([[1,0,0],\
                                    [0,-1,0],\
                                    [0,0,-1]])),np.array([0,0,np.pi]),tol))
checks.append(within_tol(rb.get_euler_angles_from_rot(\
                         np.matrix([[1,0,0],\
                                    [0,0,1],\
                                    [0,-1,0]])),np.array([0,0,-np.pi/2]),tol))
checks.append(within_tol(rb.get_euler_angles_from_rot(\
                         np.matrix([[0.903592586644424,0.150380970349009,0.401130902721456],\
                                    [0.175640606326494,0.723994214832615,-0.667070276881055],\
                                    [-0.390731128489274,0.673214631930854,0.627782800484135]])),\
                         np.array([11*np.pi/180,23*np.pi/180,47*np.pi/180]),tol))

## Unit tests for rb.skew
array0 = np.array([1,0,0])
checks.append(within_tol(rb.skew(np.array(array0)),\
                         np.matrix([[0,-array0[2],array0[1]],\
                                    [array0[2],0,-array0[0]],\
                                    [-array0[1],array0[0],0]]),tol))
array0 = array0 = np.array([0,1,0])
checks.append(within_tol(rb.skew(np.array(array0)),\
                         np.matrix([[0,-array0[2],array0[1]],\
                                    [array0[2],0,-array0[0]],\
                                    [-array0[1],array0[0],0]]),tol))
array0 = array0 = np.array([0,0,1])
checks.append(within_tol(rb.skew(np.array(array0)),\
                         np.matrix([[0,-array0[2],array0[1]],\
                                    [array0[2],0,-array0[0]],\
                                    [-array0[1],array0[0],0]]),tol))
array0 = array0 = np.array([1,2,3])
checks.append(within_tol(rb.skew(np.array(array0)),\
                         np.matrix([[0,-array0[2],array0[1]],\
                                    [array0[2],0,-array0[0]],\
                                    [-array0[1],array0[0],0]]),tol))

## Unit tests for rb.rot_from_quat
checks.append(within_tol(rb.rot_from_quat(np.array([1,0,0,0])),np.eye(3),tol))
checks.append(within_tol(rb.rot_from_quat(np.array([0,1,0,0])),\
               np.matrix([[1,0,0],[0,-1,0],[0,0,-1]]),tol))
checks.append(within_tol(rb.rot_from_quat(np.array([0,1,0,0])),\
      np.matrix([[1,0,0],[0,-1,0],[0,0,-1]]),tol))
checks.append(within_tol(rb.rot_from_quat(np.array([0,0,1,0])),\
      np.matrix([[-1,0,0],[0,1,0],[0,0,-1]]),tol))
q = [0.1,0.2,0.3,0.4]
q = q/np.linalg.norm(q)
checks.append(within_tol(rb.rot_from_quat(q),\
      np.matrix([[-0.66666667, 0.13333333, 0.73333333],\
                  [0.66666667, -0.33333333, 0.66666667],\
                  [0.33333333, 0.93333333, 0.13333333]]), tol))
    
## Unit tests for rb.quat_prod
checks.append(within_tol(rb.quat_prod(np.array([[1],[0],[0],[0]]),\
                                      np.array([[1],[0],[0],[0]])),\
                                      np.array([[1],[0],[0],[0]]),tol))
checks.append(within_tol(rb.quat_prod(np.array([[0],[1],[0],[0]]),\
                                      np.array([[0],[1],[0],[0]])),\
                                      np.array([[-1],[0],[0],[0]]),tol))
checks.append(within_tol(rb.quat_prod(np.array([[0],[0],[1],[0]]),\
                                      np.array([[0],[0],[1],[0]])),\
                                      np.array([[-1],[0],[0],[0]]),tol))
checks.append(within_tol(rb.quat_prod(np.array([[0],[0],[0],[1]]),\
                                      np.array([[0],[0],[0],[1]])),\
                                      np.array([[-1],[0],[0],[0]]),tol))
checks.append(within_tol(rb.quat_prod(np.array([[1],[0],[0],[0]]),\
                                      np.array([[0],[1],[0],[0]])),\
                                      np.array([[0],[1],[0],[0]]),tol))
checks.append(within_tol(rb.quat_prod(np.array([[1],[0],[0],[0]]),\
                                      np.array([[0],[0],[1],[0]])),\
                                      np.array([[0],[0],[1],[0]]),tol))
checks.append(within_tol(rb.quat_prod(np.array([[1],[0],[0],[0]]),\
                                      np.array([[0],[0],[0],[1]])),\
                                      np.array([[0],[0],[0],[1]]),tol))
checks.append(within_tol(rb.quat_prod(np.array([[0],[1],[0],[0]]),\
                                      np.array([[1],[0],[0],[0]])),\
                                      np.array([[0],[1],[0],[0]]),tol))
checks.append(within_tol(rb.quat_prod(np.array([[0],[0],[1],[0]]),\
                                      np.array([[1],[0],[0],[0]])),\
                                      np.array([[0],[0],[1],[0]]),tol))
checks.append(within_tol(rb.quat_prod(np.array([[0],[0],[0],[1]]),\
                                      np.array([[1],[0],[0],[0]])),\
                                      np.array([[0],[0],[0],[1]]),tol))
checks.append(within_tol(rb.quat_prod(np.array([[0],[1],[0],[0]]),\
                                      np.array([[0],[0],[1],[0]])),\
                                      np.array([[0],[0],[0],[-1]]),tol))
checks.append(within_tol(rb.quat_prod(np.array([[0],[1],[0],[0]]),\
                                      np.array([[0],[0],[0],[1]])),\
                                      np.array([[0],[0],[1],[0]]),tol))

## Unit tests for rb.quat_from_ypr
checks.append(within_tol(rb.quat_from_ypr(0, 0, 0),\
                       np.array([[1],[0],[0],[0]]),tol))
checks.append(within_tol(rb.quat_from_ypr(np.pi/2, 0, 0),\
                       np.array([[0.707106781186548],[0],[0],[0.707106781186547]]),tol))
checks.append(within_tol(rb.quat_from_ypr(-np.pi/2, 0, 0),\
                       np.array([[0.707106781186548],[0],[0],[-0.707106781186547]]),tol))
checks.append(within_tol(rb.quat_from_ypr(0, np.pi/2, 0),\
                       np.array([[0.707106781186548],[0],[0.707106781186547],[0]]),tol))
checks.append(within_tol(rb.quat_from_ypr(0, -np.pi/2, 0),\
                       np.array([[0.707106781186548],[0],[-0.707106781186547],[0]]),tol))
checks.append(within_tol(rb.quat_from_ypr(0, 0, np.pi/2 ),\
                       np.array([[0.707106781186548],[0.707106781186547],[0],[0]]),tol))
checks.append(within_tol(rb.quat_from_ypr(0, 0, -np.pi/2 ),\
                       np.array([[0.707106781186548],[-0.707106781186547],[0],[0]]),tol))
checks.append(within_tol(rb.quat_from_ypr(np.pi/180*47,np.pi/180*15,np.pi/180*6),\
                       np.array([[0.910692391306739],[-0.004391258543109],\
                                 [0.140226691736355],[0.388531285984923]]),tol))

## Unit tests for rb.RigidBody init
m = 1
J = np.eye(3)
x0 = np.array([0.0,0.0,0.0, 0.0,0.0,0.0, 1.0,0.0,0.0,0.0, 0.0,0.0,0.0])
x0 = x0.reshape(x0.shape[0],1)
drone = rb.RigidBody(m,J,x0)

checks.append(within_tol(m,drone.m,tol))
checks.append(within_tol(J,drone.J,tol))
checks.append(within_tol(x0,drone.x,tol))

m = uav.m
J = uav.J
x0 = np.array([1.0,2.0,3.0, 4.0,5.0,6.0, 0.0,1/np.sqrt(2),1/np.sqrt(2),0.0, 7.0,8.0,9.0])
x0 = x0.reshape(x0.shape[0],1)
drone = rb.RigidBody(m,J,x0)

checks.append(within_tol(m,drone.m,tol))
checks.append(within_tol(J,drone.J,tol))
checks.append(within_tol(x0,drone.x,tol))

## Unit tests for rb.RigidBody.f, change in inertial position
# No velocities and no accels 
t0 = 0
m = 1
J = np.eye(3)
x0 = np.array([0.0,0.0,0.0, 0.0,0.0,0.0, 1.0,0.0,0.0,0.0, 0.0,0.0,0.0])
x0 = x0.reshape(x0.shape[0],1)
u0 = np.array([0.0,0.0,0.0, 0.0,0.0,0.0])
u0 = u0.reshape(u0.shape[0],1)
drone = rb.RigidBody(m,J,x0)
expected_result = np.array([0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0])
expected_result = expected_result.reshape(expected_result.shape[0],1)
checks.append(within_tol(drone.f(t0,drone.x,u0),expected_result,tol))
# Same but show time invarience
t0 = 10
drone = rb.RigidBody(m,J,x0)
expected_result = np.array([0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0])
expected_result = expected_result.reshape(expected_result.shape[0],1)
checks.append(within_tol(drone.f(t0,drone.x,u0),expected_result,tol))
# pn = 1, 0 euler angles, no velocites, forces, or moments
t0 = 0
yaw0 = np.deg2rad(0)
pitch0 = np.deg2rad(0)
roll0 = np.deg2rad(0)
quat0 = rb.quat_from_ypr(yaw0,pitch0,roll0)
x0 = np.array([1.0,0.0,0.0, 0.0,0.0,0.0, quat0[0],quat0[1],quat0[2],quat0[3], 0.0,0.0,0.0])
x0 = x0.reshape(x0.shape[0],1)
drone = rb.RigidBody(m,J,x0)
expected_result = np.array([0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0])
expected_result = expected_result.reshape(expected_result.shape[0],1)
checks.append(within_tol(drone.f(t0,drone.x,u0),expected_result,tol))
# pe = 1, 0 euler angles, no velocites, forces, or moments
x0 = np.array([0.0,1.0,0.0, 0.0,0.0,0.0, quat0[0],quat0[1],quat0[2],quat0[3], 0.0,0.0,0.0])
x0 = x0.reshape(x0.shape[0],1)
drone = rb.RigidBody(m,J,x0)
expected_result = np.array([0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0])
expected_result = expected_result.reshape(expected_result.shape[0],1)
checks.append(within_tol(drone.f(t0,drone.x,u0),expected_result,tol))
# pd = 1, 0 euler angles, no velocites, forces, or moments
x0 = np.array([0.0,0.0,1.0, 0.0,0.0,0.0, quat0[0],quat0[1],quat0[2],quat0[3], 0.0,0.0,0.0])
x0 = x0.reshape(x0.shape[0],1)
drone = rb.RigidBody(m,J,x0)
expected_result = np.array([0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0])
expected_result = expected_result.reshape(expected_result.shape[0],1)
checks.append(within_tol(drone.f(t0,drone.x,u0),expected_result,tol))
# u = 1, 0 euler angles, no forces or moments
x0 = np.array([0.0,0.0,0.0, 1.0,0.0,0.0, quat0[0],quat0[1],quat0[2],quat0[3], 0.0,0.0,0.0])
x0 = x0.reshape(x0.shape[0],1)
drone = rb.RigidBody(m,J,x0)
expected_result = np.array([x0[3],0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0])
expected_result = expected_result.reshape(expected_result.shape[0],1)
checks.append(within_tol(drone.f(t0,drone.x,u0),expected_result,tol))
# v = 1, 0 euler angles, no forces or moments
x0 = np.array([0.0,0.0,0.0, 0.0,1.0,0.0, quat0[0],quat0[1],quat0[2],quat0[3], 0.0,0.0,0.0])
x0 = x0.reshape(x0.shape[0],1)
drone = rb.RigidBody(m,J,x0)
expected_result = np.array([0.0,x0[4],0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0])
expected_result = expected_result.reshape(expected_result.shape[0],1)
checks.append(within_tol(drone.f(t0,drone.x,u0),expected_result,tol))
# w = 1, 0 euler angles, no forces or moments
x0 = np.array([0.0,0.0,0.0, 0.0,0.0,1.0, quat0[0],quat0[1],quat0[2],quat0[3], 0.0,0.0,0.0])
x0 = x0.reshape(x0.shape[0],1)
drone = rb.RigidBody(m,J,x0)
expected_result = np.array([0.0,0.0,x0[5], 0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0])
expected_result = expected_result.reshape(expected_result.shape[0],1)
checks.append(within_tol(drone.f(t0,drone.x,u0),expected_result,tol))
# u = 5, 0 euler angles, no forces or moments
x0 = np.array([0.0,0.0,0.0, 5.0,0.0,0.0, quat0[0],quat0[1],quat0[2],quat0[3], 0.0,0.0,0.0])
x0 = x0.reshape(x0.shape[0],1)
drone = rb.RigidBody(m,J,x0)
expected_result = np.array([x0[3],0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0])
expected_result = expected_result.reshape(expected_result.shape[0],1)
checks.append(within_tol(drone.f(t0,drone.x,u0),expected_result,tol))
# v = 10, 0 euler angles, no forces or moments
x0 = np.array([0.0,0.0,0.0, 0.0,10.0,0.0, quat0[0],quat0[1],quat0[2],quat0[3], 0.0,0.0,0.0])
x0 = x0.reshape(x0.shape[0],1)
drone = rb.RigidBody(m,J,x0)
expected_result = np.array([0.0,x0[4],0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0])
expected_result = expected_result.reshape(expected_result.shape[0],1)
checks.append(within_tol(drone.f(t0,drone.x,u0),expected_result,tol))
# w = 15, 0 euler angles, no forces or moments
x0 = np.array([0.0,0.0,0.0, 0.0,0.0,15.0, quat0[0],quat0[1],quat0[2],quat0[3], 0.0,0.0,0.0])
x0 = x0.reshape(x0.shape[0],1)
drone = rb.RigidBody(m,J,x0)
expected_result = np.array([0.0,0.0,x0[5], 0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0])
expected_result = expected_result.reshape(expected_result.shape[0],1)
checks.append(within_tol(drone.f(t0,drone.x,u0),expected_result,tol))
# u = 1,v = 2,w = 3 yaw 90 deg, no forces or moments
yaw0 = np.deg2rad(90)
pitch0 = np.deg2rad(0)
roll0 = np.deg2rad(0)
quat0 = rb.quat_from_ypr(yaw0,pitch0,roll0)
x0 = np.array([0.0,0.0,0.0, 1.0,2.0,3.0, quat0[0],quat0[1],quat0[2],quat0[3], 0.0,0.0,0.0])
x0 = x0.reshape(x0.shape[0],1)
drone = rb.RigidBody(m,J,x0)
expected_result = np.array([-x0[4],x0[3],x0[5], 0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0])
expected_result = expected_result.reshape(expected_result.shape[0],1)
checks.append(within_tol(drone.f(t0,drone.x,u0),expected_result,tol))
# u = 1,v = 2,w = 3 yaw 180 deg, no forces or moments
yaw0 = np.deg2rad(180)
pitch0 = np.deg2rad(0)
roll0 = np.deg2rad(0)
quat0 = rb.quat_from_ypr(yaw0,pitch0,roll0)
x0 = np.array([0.0,0.0,0.0, 1.0,2.0,3.0, quat0[0],quat0[1],quat0[2],quat0[3], 0.0,0.0,0.0])
x0 = x0.reshape(x0.shape[0],1)
drone = rb.RigidBody(m,J,x0)
expected_result = np.array([-x0[3],-x0[4],x0[5], 0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0])
expected_result = expected_result.reshape(expected_result.shape[0],1)
checks.append(within_tol(drone.f(t0,drone.x,u0),expected_result,tol))
# u = 1,v = 2,w = 3 yaw -90 deg, no forces or moments
yaw0 = np.deg2rad(-90)
pitch0 = np.deg2rad(0)
roll0 = np.deg2rad(0)
quat0 = rb.quat_from_ypr(yaw0,pitch0,roll0)
x0 = np.array([0.0,0.0,0.0, 1.0,2.0,3.0, quat0[0],quat0[1],quat0[2],quat0[3], 0.0,0.0,0.0])
x0 = x0.reshape(x0.shape[0],1)
drone = rb.RigidBody(m,J,x0)
expected_result = np.array([x0[4],-x0[3],x0[5], 0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0])
expected_result = expected_result.reshape(expected_result.shape[0],1)
checks.append(within_tol(drone.f(t0,drone.x,u0),expected_result,tol))
# u = 1,v = 2,w = 3 pitch 90 deg, no forces or moments
yaw0 = np.deg2rad(0)
pitch0 = np.deg2rad(90)
roll0 = np.deg2rad(0)
quat0 = rb.quat_from_ypr(yaw0,pitch0,roll0)
x0 = np.array([0.0,0.0,0.0, 1.0,2.0,3.0, quat0[0],quat0[1],quat0[2],quat0[3], 0.0,0.0,0.0])
x0 = x0.reshape(x0.shape[0],1)
drone = rb.RigidBody(m,J,x0)
expected_result = np.array([x0[5],x0[4],-x0[3], 0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0])
expected_result = expected_result.reshape(expected_result.shape[0],1)
checks.append(within_tol(drone.f(t0,drone.x,u0),expected_result,tol))
# u = 1,v = 2,w = 3 pitch 180 deg, no forces or moments
yaw0 = np.deg2rad(0)
pitch0 = np.deg2rad(180)
roll0 = np.deg2rad(0)
quat0 = rb.quat_from_ypr(yaw0,pitch0,roll0)
x0 = np.array([0.0,0.0,0.0, 1.0,2.0,3.0, quat0[0],quat0[1],quat0[2],quat0[3], 0.0,0.0,0.0])
x0 = x0.reshape(x0.shape[0],1)
drone = rb.RigidBody(m,J,x0)
expected_result = np.array([-x0[3],x0[4],-x0[5], 0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0])
expected_result = expected_result.reshape(expected_result.shape[0],1)
checks.append(within_tol(drone.f(t0,drone.x,u0),expected_result,tol))
# u = 1,v = 2,w = 3 pitch -90 deg, no forces or moments
yaw0 = np.deg2rad(0)
pitch0 = np.deg2rad(-90)
roll0 = np.deg2rad(0)
quat0 = rb.quat_from_ypr(yaw0,pitch0,roll0)
x0 = np.array([0.0,0.0,0.0, 1.0,2.0,3.0, quat0[0],quat0[1],quat0[2],quat0[3], 0.0,0.0,0.0])
x0 = x0.reshape(x0.shape[0],1)
drone = rb.RigidBody(m,J,x0)
expected_result = np.array([-x0[5],x0[4],x0[3], 0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0])
expected_result = expected_result.reshape(expected_result.shape[0],1)
checks.append(within_tol(drone.f(t0,drone.x,u0),expected_result,tol))
# u = 1,v = 2,w = 3 roll 90 deg, no forces or moments
yaw0 = np.deg2rad(0)
pitch0 = np.deg2rad(0)
roll0 = np.deg2rad(90)
quat0 = rb.quat_from_ypr(yaw0,pitch0,roll0)
x0 = np.array([0.0,0.0,0.0, 1.0,2.0,3.0, quat0[0],quat0[1],quat0[2],quat0[3], 0.0,0.0,0.0])
x0 = x0.reshape(x0.shape[0],1)
drone = rb.RigidBody(m,J,x0)
expected_result = np.array([x0[3],-x0[5],x0[4], 0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0])
expected_result = expected_result.reshape(expected_result.shape[0],1)
checks.append(within_tol(drone.f(t0,drone.x,u0),expected_result,tol))
# u = 1,v = 2,w = 3 roll 180 deg, no forces or moments
yaw0 = np.deg2rad(0)
pitch0 = np.deg2rad(0)
roll0 = np.deg2rad(180)
quat0 = rb.quat_from_ypr(yaw0,pitch0,roll0)
x0 = np.array([0.0,0.0,0.0, 1.0,2.0,3.0, quat0[0],quat0[1],quat0[2],quat0[3], 0.0,0.0,0.0])
x0 = x0.reshape(x0.shape[0],1)
drone = rb.RigidBody(m,J,x0)
expected_result = np.array([x0[3],-x0[4],-x0[5], 0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0])
expected_result = expected_result.reshape(expected_result.shape[0],1)
checks.append(within_tol(drone.f(t0,drone.x,u0),expected_result,tol))
# u = 1,v = 2,w = 3 roll -90 deg, no forces or moments
yaw0 = np.deg2rad(0)
pitch0 = np.deg2rad(0)
roll0 = np.deg2rad(-90)
quat0 = rb.quat_from_ypr(yaw0,pitch0,roll0)
x0 = np.array([0.0,0.0,0.0, 1.0,2.0,3.0, quat0[0],quat0[1],quat0[2],quat0[3], 0.0,0.0,0.0])
x0 = x0.reshape(x0.shape[0],1)
drone = rb.RigidBody(m,J,x0)
expected_result = np.array([x0[3],x0[5],-x0[4], 0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0])
expected_result = expected_result.reshape(expected_result.shape[0],1)
checks.append(within_tol(drone.f(t0,drone.x,u0),expected_result,tol))

## Unit tests for rb.RigidBody.f, change in velocities
#No velocity no forces
t0 = 0
m = 5
J = np.eye(3)
x0 = np.array([0.0,0.0,0.0, 0.0,0.0,0.0, 1.0,0.0,0.0,0.0, 0.0,0.0,0.0])
x0 = x0.reshape(x0.shape[0],1)
u0 = np.array([0.0,0.0,0.0, 0.0,0.0,0.0])
u0 = u0.reshape(u0.shape[0],1)
drone = rb.RigidBody(m,J,x0)
expected_result = np.array([0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0])
expected_result = expected_result.reshape(expected_result.shape[0],1)
checks.append(within_tol(drone.f(t0,drone.x,u0),expected_result,tol))
#fx = m, no velocities
x0 = np.array([0.0,0.0,0.0, 0.0,0.0,0.0, 1.0,0.0,0.0,0.0, 0.0,0.0,0.0])
x0 = x0.reshape(x0.shape[0],1)
u0 = np.array([m,0.0,0.0, 0.0,0.0,0.0])
u0 = u0.reshape(u0.shape[0],1)
drone = rb.RigidBody(m,J,x0)
expected_result = np.array([0.0,0.0,0.0, 1.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0])
expected_result = expected_result.reshape(expected_result.shape[0],1)
checks.append(within_tol(drone.f(t0,drone.x,u0),expected_result,tol))
#fx = 3m, no velocities
x0 = np.array([0.0,0.0,0.0, 0.0,0.0,0.0, 1.0,0.0,0.0,0.0, 0.0,0.0,0.0])
x0 = x0.reshape(x0.shape[0],1)
u0 = np.array([3*m,0.0,0.0, 0.0,0.0,0.0])
u0 = u0.reshape(u0.shape[0],1)
drone = rb.RigidBody(m,J,x0)
expected_result = np.array([0.0,0.0,0.0, 3.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0])
expected_result = expected_result.reshape(expected_result.shape[0],1)
checks.append(within_tol(drone.f(t0,drone.x,u0),expected_result,tol))
#fx = -5m, no velocities
x0 = np.array([0.0,0.0,0.0, 0.0,0.0,0.0, 1.0,0.0,0.0,0.0, 0.0,0.0,0.0])
x0 = x0.reshape(x0.shape[0],1)
u0 = np.array([-5*m,0.0,0.0, 0.0,0.0,0.0])
u0 = u0.reshape(u0.shape[0],1)
drone = rb.RigidBody(m,J,x0)
expected_result = np.array([0.0,0.0,0.0, -5.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0])
expected_result = expected_result.reshape(expected_result.shape[0],1)
checks.append(within_tol(drone.f(t0,drone.x,u0),expected_result,tol))
#fy = m, no velocities
x0 = np.array([0.0,0.0,0.0, 0.0,0.0,0.0, 1.0,0.0,0.0,0.0, 0.0,0.0,0.0])
x0 = x0.reshape(x0.shape[0],1)
u0 = np.array([0.0,m,0.0, 0.0,0.0,0.0])
u0 = u0.reshape(u0.shape[0],1)
drone = rb.RigidBody(m,J,x0)
expected_result = np.array([0.0,0.0,0.0, 0.0,1.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0])
expected_result = expected_result.reshape(expected_result.shape[0],1)
checks.append(within_tol(drone.f(t0,drone.x,u0),expected_result,tol))
#fy = 3m, no velocities
x0 = np.array([0.0,0.0,0.0, 0.0,0.0,0.0, 1.0,0.0,0.0,0.0, 0.0,0.0,0.0])
x0 = x0.reshape(x0.shape[0],1)
u0 = np.array([0.0,3*m,0.0, 0.0,0.0,0.0])
u0 = u0.reshape(u0.shape[0],1)
drone = rb.RigidBody(m,J,x0)
expected_result = np.array([0.0,0.0,0.0, 0.0,3.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0])
expected_result = expected_result.reshape(expected_result.shape[0],1)
checks.append(within_tol(drone.f(t0,drone.x,u0),expected_result,tol))
#fy = -5m, no velocities
x0 = np.array([0.0,0.0,0.0, 0.0,0.0,0.0, 1.0,0.0,0.0,0.0, 0.0,0.0,0.0])
x0 = x0.reshape(x0.shape[0],1)
u0 = np.array([0.0,-5*m,0.0, 0.0,0.0,0.0])
u0 = u0.reshape(u0.shape[0],1)
drone = rb.RigidBody(m,J,x0)
expected_result = np.array([0.0,0.0,0.0, 0.0,-5.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0])
expected_result = expected_result.reshape(expected_result.shape[0],1)
checks.append(within_tol(drone.f(t0,drone.x,u0),expected_result,tol))
#fz = m, no velocities
x0 = np.array([0.0,0.0,0.0, 0.0,0.0,0.0, 1.0,0.0,0.0,0.0, 0.0,0.0,0.0])
x0 = x0.reshape(x0.shape[0],1)
u0 = np.array([0.0,0.0,m, 0.0,0.0,0.0])
u0 = u0.reshape(u0.shape[0],1)
drone = rb.RigidBody(m,J,x0)
expected_result = np.array([0.0,0.0,0.0, 0.0,0.0,1.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0])
expected_result = expected_result.reshape(expected_result.shape[0],1)
checks.append(within_tol(drone.f(t0,drone.x,u0),expected_result,tol))
#fz = 3m, no velocities
x0 = np.array([0.0,0.0,0.0, 0.0,0.0,0.0, 1.0,0.0,0.0,0.0, 0.0,0.0,0.0])
x0 = x0.reshape(x0.shape[0],1)
u0 = np.array([0.0,0.0,3*m, 0.0,0.0,0.0])
u0 = u0.reshape(u0.shape[0],1)
drone = rb.RigidBody(m,J,x0)
expected_result = np.array([0.0,0.0,0.0, 0.0,0.0,3.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0])
expected_result = expected_result.reshape(expected_result.shape[0],1)
checks.append(within_tol(drone.f(t0,drone.x,u0),expected_result,tol))
#fz = -5m, no velocities
x0 = np.array([0.0,0.0,0.0, 0.0,0.0,0.0, 1.0,0.0,0.0,0.0, 0.0,0.0,0.0])
x0 = x0.reshape(x0.shape[0],1)
u0 = np.array([0.0,0.0,-5*m, 0.0,0.0,0.0])
u0 = u0.reshape(u0.shape[0],1)
drone = rb.RigidBody(m,J,x0)
expected_result = np.array([0.0,0.0,0.0, 0.0,0.0,-5.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0])
expected_result = expected_result.reshape(expected_result.shape[0],1)
checks.append(within_tol(drone.f(t0,drone.x,u0),expected_result,tol))
#No force applied, [u,v,w] = [0,1,0], [p,q,r] = [0,0,1]
x0 = np.array([0.0,0.0,0.0, 0.0,1.0,0.0, 1.0,0.0,0.0,0.0, 0.0,0.0,1.0])
x0 = x0.reshape(x0.shape[0],1)
u0 = np.array([0.0,0.0,0, 0.0,0.0,0.0])
u0 = u0.reshape(u0.shape[0],1)
drone = rb.RigidBody(m,J,x0)
expected_result = np.array([1.0,0.0,0.0])
expected_result = expected_result.reshape(expected_result.shape[0],1)
result = drone.f(t0,drone.x,u0)[3:6]
checks.append(within_tol(result,expected_result,tol))
#No force applied, [u,v,w] = [0,0,1], [p,q,r] = [0,1,0]
x0 = np.array([0.0,0.0,0.0, 0.0,0.0,1.0, 1.0,0.0,0.0,0.0, 0.0,1.0,0.0])
x0 = x0.reshape(x0.shape[0],1)
u0 = np.array([0.0,0.0,0, 0.0,0.0,0.0])
u0 = u0.reshape(u0.shape[0],1)
drone = rb.RigidBody(m,J,x0)
expected_result = np.array([-1.0,0.0,0.0])
expected_result = expected_result.reshape(expected_result.shape[0],1)
result = drone.f(t0,drone.x,u0)[3:6]
checks.append(within_tol(result,expected_result,tol))
#No force applied, [u,v,w] = [0,0,1], [p,q,r] = [1,0,0]
x0 = np.array([0.0,0.0,0.0, 0.0,0.0,1.0, 1.0,0.0,0.0,0.0, 1.0,0.0,0.0])
x0 = x0.reshape(x0.shape[0],1)
u0 = np.array([0.0,0.0,0, 0.0,0.0,0.0])
u0 = u0.reshape(u0.shape[0],1)
drone = rb.RigidBody(m,J,x0)
expected_result = np.array([0.0,1.0,0.0])
expected_result = expected_result.reshape(expected_result.shape[0],1)
result = drone.f(t0,drone.x,u0)[3:6]
checks.append(within_tol(result,expected_result,tol))
#No force applied, [u,v,w] = [1,0,0], [p,q,r] = [0,0,1]
x0 = np.array([0.0,0.0,0.0, 1.0,0.0,0.0, 1.0,0.0,0.0,0.0, 0.0,0.0,1.0])
x0 = x0.reshape(x0.shape[0],1)
u0 = np.array([0.0,0.0,0, 0.0,0.0,0.0])
u0 = u0.reshape(u0.shape[0],1)
drone = rb.RigidBody(m,J,x0)
expected_result = np.array([0.0,-1.0,0.0])
expected_result = expected_result.reshape(expected_result.shape[0],1)
result = drone.f(t0,drone.x,u0)[3:6]
checks.append(within_tol(result,expected_result,tol))
#No force applied, [u,v,w] = [1,0,0], [p,q,r] = [0,1,0]
x0 = np.array([0.0,0.0,0.0, 1.0,0.0,0.0, 1.0,0.0,0.0,0.0, 0.0,1.0,0.0])
x0 = x0.reshape(x0.shape[0],1)
u0 = np.array([0.0,0.0,0, 0.0,0.0,0.0])
u0 = u0.reshape(u0.shape[0],1)
drone = rb.RigidBody(m,J,x0)
expected_result = np.array([0.0,0.0,1.0])
expected_result = expected_result.reshape(expected_result.shape[0],1)
result = drone.f(t0,drone.x,u0)[3:6]
checks.append(within_tol(result,expected_result,tol))
#No force applied, [u,v,w] = [0,1,0], [p,q,r] = [1,0,0]
x0 = np.array([0.0,0.0,0.0, 0.0,1.0,0.0, 1.0,0.0,0.0,0.0, 1.0,0.0,0.0])
x0 = x0.reshape(x0.shape[0],1)
u0 = np.array([0.0,0.0,0, 0.0,0.0,0.0])
u0 = u0.reshape(u0.shape[0],1)
drone = rb.RigidBody(m,J,x0)
expected_result = np.array([0.0,0.0,-1.0])
expected_result = expected_result.reshape(expected_result.shape[0],1)
result = drone.f(t0,drone.x,u0)[3:6]
checks.append(within_tol(result,expected_result,tol))
#No force applied, [u,v,w] = [1,2,3], [p,q,r] = [4,5,6]
x0 = np.array([0.0,0.0,0.0, 1.0,2.0,3.0, 1.0,0.0,0.0,0.0, 4.0,5.0,6.0])
x0 = x0.reshape(x0.shape[0],1)
u0 = np.array([0.0,0.0,0, 0.0,0.0,0.0])
u0 = u0.reshape(u0.shape[0],1)
drone = rb.RigidBody(m,J,x0)
expected_result = np.array([-3.0,6.0,-3.0])
expected_result = expected_result.reshape(expected_result.shape[0],1)
result = drone.f(t0,drone.x,u0)[3:6]
checks.append(within_tol(result,expected_result,tol))

## Unit tests for rb.RigidBody.f, change in quaternion
# No angular velocities
t0 = 0
m = 5
J = np.eye(3)
x0 = np.array([0.0,0.0,0.0, 0.0,0.0,0.0, 1.0,0.0,0.0,0.0, 0.0,0.0,0.0])
x0 = x0.reshape(x0.shape[0],1)
u0 = np.array([0.0,0.0,0.0, 0.0,0.0,0.0])
u0 = u0.reshape(u0.shape[0],1)
drone = rb.RigidBody(m,J,x0)
expected_result = np.array([0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0])
expected_result = expected_result.reshape(expected_result.shape[0],1)
checks.append(within_tol(drone.f(t0,drone.x,u0),expected_result,tol))
# quat = [1,0,0,0], p = 1
x0 = np.array([0.0,0.0,0.0, 0.0,0.0,0.0, 1.0,0.0,0.0,0.0, 1.0,0.0,0.0])
x0 = x0.reshape(x0.shape[0],1)
drone = rb.RigidBody(m,J,x0)
expected_result = np.array([0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.5,0.0,0.0, 0.0,0.0,0.0])
expected_result = expected_result.reshape(expected_result.shape[0],1)
checks.append(within_tol(drone.f(t0,drone.x,u0),expected_result,tol))
# quat = [1,0,0,0], p = -1
x0 = np.array([0.0,0.0,0.0, 0.0,0.0,0.0, 1.0,0.0,0.0,0.0, -1.0,0.0,0.0])
x0 = x0.reshape(x0.shape[0],1)
drone = rb.RigidBody(m,J,x0)
expected_result = np.array([0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,-0.5,0.0,0.0, 0.0,0.0,0.0])
expected_result = expected_result.reshape(expected_result.shape[0],1)
checks.append(within_tol(drone.f(t0,drone.x,u0),expected_result,tol))
# quat = [1,0,0,0], q = 1
x0 = np.array([0.0,0.0,0.0, 0.0,0.0,0.0, 1.0,0.0,0.0,0.0, 0.0,1.0,0.0])
x0 = x0.reshape(x0.shape[0],1)
drone = rb.RigidBody(m,J,x0)
expected_result = np.array([0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.5,0.0, 0.0,0.0,0.0])
expected_result = expected_result.reshape(expected_result.shape[0],1)
checks.append(within_tol(drone.f(t0,drone.x,u0),expected_result,tol))
# quat = [1,0,0,0], q = -1
x0 = np.array([0.0,0.0,0.0, 0.0,0.0,0.0, 1.0,0.0,0.0,0.0, 0.0,-1.0,0.0])
x0 = x0.reshape(x0.shape[0],1)
drone = rb.RigidBody(m,J,x0)
expected_result = np.array([0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,-0.5,0.0, 0.0,0.0,0.0])
expected_result = expected_result.reshape(expected_result.shape[0],1)
checks.append(within_tol(drone.f(t0,drone.x,u0),expected_result,tol))
# quat = [1,0,0,0], r = 1
x0 = np.array([0.0,0.0,0.0, 0.0,0.0,0.0, 1.0,0.0,0.0,0.0, 0.0,0.0,1.0])
x0 = x0.reshape(x0.shape[0],1)
drone = rb.RigidBody(m,J,x0)
expected_result = np.array([0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,0.5, 0.0,0.0,0.0])
expected_result = expected_result.reshape(expected_result.shape[0],1)
checks.append(within_tol(drone.f(t0,drone.x,u0),expected_result,tol))
# quat = [1,0,0,0], r = -1
x0 = np.array([0.0,0.0,0.0, 0.0,0.0,0.0, 1.0,0.0,0.0,0.0, 0.0,0.0,-1.0])
x0 = x0.reshape(x0.shape[0],1)
drone = rb.RigidBody(m,J,x0)
expected_result = np.array([0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,-0.5, 0.0,0.0,0.0])
expected_result = expected_result.reshape(expected_result.shape[0],1)
checks.append(within_tol(drone.f(t0,drone.x,u0),expected_result,tol))
# quat = [1,0,0,0], [p,q,r] = [1,2,3]
x0 = np.array([0.0,0.0,0.0, 0.0,0.0,0.0, 1.0,0.0,0.0,0.0, 1.0,2.0,3.0])
x0 = x0.reshape(x0.shape[0],1)
drone = rb.RigidBody(m,J,x0)
expected_result = np.array([0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.5,1.0,1.5, 0.0,0.0,0.0])
expected_result = expected_result.reshape(expected_result.shape[0],1)
checks.append(within_tol(drone.f(t0,drone.x,u0),expected_result,tol))
# Random inputs for quat and omega
t0 = 0
yaw0 = np.deg2rad(-52)
pitch0 = np.deg2rad(16)
roll0 = np.deg2rad(-4)
quat0 = rb.quat_from_ypr(yaw0,pitch0,roll0)
x0 = np.array([1.0,0.0,0.0, 0.0,0.0,0.0, quat0[0],quat0[1],quat0[2],quat0[3], -6.0,-7.0,3.0])
x0 = x0.reshape(x0.shape[0],1)
drone = rb.RigidBody(m,J,x0)
expected_result = np.array([0.0,0.0,0.0, 0.0,0.0,0.0,\
                            1.224509128885703, -3.967821947569283, -1.877159358071763, 1.653251078955249,\
                                0.0,0.0,0.0])
expected_result = expected_result.reshape(expected_result.shape[0],1)
checks.append(within_tol(drone.f(t0,drone.x,u0),expected_result,tol))

## Unit tests for rb.RigidBody.f, change in angular velocities
# Stationary, no torques
t0 = 0
m = 5
x0 = np.array([0.0,0.0,0.0, 0.0,0.0,0.0, 1.0,0.0,0.0,0.0, 0.0,0.0,0.0])
x0 = x0.reshape(x0.shape[0],1)
u0 = np.array([0.0,0.0,0.0, 0.0,0.0,0.0])
u0 = u0.reshape(u0.shape[0],1)
drone = rb.RigidBody(m,J,x0)
expected_result = np.array([0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0])
expected_result = expected_result.reshape(expected_result.shape[0],1)
checks.append(within_tol(drone.f(t0,drone.x,u0),expected_result,tol))
# mx = J[0,0]
u0 = np.array([0.0,0.0,0.0, J[0,0],0.0,0.0])
u0 = u0.reshape(u0.shape[0],1)
drone = rb.RigidBody(m,J,x0)
expected_result = np.array([0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 1.0,0.0,0.0])
expected_result = expected_result.reshape(expected_result.shape[0],1)
checks.append(within_tol(drone.f(t0,drone.x,u0),expected_result,tol))
# mx = 3*J[0,0]
u0 = np.array([0.0,0.0,0.0, 3*J[0,0],0.0,0.0])
u0 = u0.reshape(u0.shape[0],1)
drone = rb.RigidBody(m,J,x0)
expected_result = np.array([0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 3.0,0.0,0.0])
expected_result = expected_result.reshape(expected_result.shape[0],1)
checks.append(within_tol(drone.f(t0,drone.x,u0),expected_result,tol))
# mx = -5*J[0,0]
u0 = np.array([0.0,0.0,0.0, -5*J[0,0],0.0,0.0])
u0 = u0.reshape(u0.shape[0],1)
drone = rb.RigidBody(m,J,x0)
expected_result = np.array([0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,0.0, -5.0,0.0,0.0])
expected_result = expected_result.reshape(expected_result.shape[0],1)
checks.append(within_tol(drone.f(t0,drone.x,u0),expected_result,tol))
# my = J[1,1]
u0 = np.array([0.0,0.0,0.0, 0.0,J[1,1],0.0])
u0 = u0.reshape(u0.shape[0],1)
drone = rb.RigidBody(m,J,x0)
expected_result = np.array([0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,1.0,0.0])
expected_result = expected_result.reshape(expected_result.shape[0],1)
checks.append(within_tol(drone.f(t0,drone.x,u0),expected_result,tol))
# my = 3*J[1,1]
u0 = np.array([0.0,0.0,0.0, 0.0,3*J[1,1],0.0])
u0 = u0.reshape(u0.shape[0],1)
drone = rb.RigidBody(m,J,x0)
expected_result = np.array([0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,3.0,0.0])
expected_result = expected_result.reshape(expected_result.shape[0],1)
checks.append(within_tol(drone.f(t0,drone.x,u0),expected_result,tol))
# my = -5*J[1,1]
u0 = np.array([0.0,0.0,0.0, 0.0,-5*J[0,0],0.0])
u0 = u0.reshape(u0.shape[0],1)
drone = rb.RigidBody(m,J,x0)
expected_result = np.array([0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,-5.0,0.0])
expected_result = expected_result.reshape(expected_result.shape[0],1)
checks.append(within_tol(drone.f(t0,drone.x,u0),expected_result,tol))
# mz = J[2,2]
u0 = np.array([0.0,0.0,0.0, 0.0,0.0,J[2,2]])
u0 = u0.reshape(u0.shape[0],1)
drone = rb.RigidBody(m,J,x0)
expected_result = np.array([0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,1.0])
expected_result = expected_result.reshape(expected_result.shape[0],1)
checks.append(within_tol(drone.f(t0,drone.x,u0),expected_result,tol))
# mz = 3*J[2,2]
u0 = np.array([0.0,0.0,0.0, 0.0,0.0,3*J[2,2]])
u0 = u0.reshape(u0.shape[0],1)
drone = rb.RigidBody(m,J,x0)
expected_result = np.array([0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,3.0])
expected_result = expected_result.reshape(expected_result.shape[0],1)
checks.append(within_tol(drone.f(t0,drone.x,u0),expected_result,tol))
# mz = -5*J[2,2]
u0 = np.array([0.0,0.0,0.0, 0.0,0.0,-5*J[2,2]])
u0 = u0.reshape(u0.shape[0],1)
drone = rb.RigidBody(m,J,x0)
expected_result = np.array([0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,-5.0])
expected_result = expected_result.reshape(expected_result.shape[0],1)
checks.append(within_tol(drone.f(t0,drone.x,u0),expected_result,tol))
# No moments, non-diagnol J matrix
J = np.matrix([[1,0,-0.1],[0,1,0],[-0.1,0,1]])
x0 = np.array([0.0,0.0,0.0, 0.0,0.0,0.0, 1.0,0.0,0.0,0.0, 1.0,0.0,0.0])
x0 = x0.reshape(x0.shape[0],1)
u0 = np.array([0.0,0.0,0.0, 0.0,0.0,0.0])
u0 = u0.reshape(u0.shape[0],1)
drone = rb.RigidBody(m,J,x0)
expected_result = np.array([[0.0],[J[0,2]*x0[10]**2],[0.0]])
result = drone.f(t0,drone.x,u0)[10:]
checks.append(within_tol(result,expected_result,tol))
# No moments, different J matrix, different speed
J = np.matrix([[1,0,-0.2],[0,1,0],[-0.2,0,1]])
x0 = np.array([0.0,0.0,0.0, 0.0,0.0,0.0, 1.0,0.0,0.0,0.0, 2.0,0.0,0.0])
x0 = x0.reshape(x0.shape[0],1)
u0 = np.array([0.0,0.0,0.0, 0.0,0.0,0.0])
u0 = u0.reshape(u0.shape[0],1)
drone = rb.RigidBody(m,J,x0)
expected_result = np.array([[0.0],[J[0,2]*x0[10]**2],[0.0]])
result = drone.f(t0,drone.x,u0)[10:]
checks.append(within_tol(result,expected_result,tol))

    
## Print True if all checks pass
print(all(checks))