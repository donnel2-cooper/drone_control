## Unit tests for rigid_body functions
import rigid_body as rb
import numpy as np

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
                                      np.array([[0],[0],[0],[1]]),tol))
checks.append(within_tol(rb.quat_prod(np.array([[0],[1],[0],[0]]),\
                                      np.array([[0],[0],[0],[1]])),\
                                      np.array([[0],[0],[-1],[0]]),tol))
checks.append(within_tol(rb.quat_prod(rb.quat_from_ypr(14,71,35),\
                                      rb.quat_from_ypr(87,36,13)),\
                                      np.array([[0.45773057],[-0.25990832],[0.08678468],[-0.84581252]]),\
                                      tol))

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
    

    
## Print True if all checks pass
print(all(checks))