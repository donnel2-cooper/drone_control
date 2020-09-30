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
    
## Unit tests for rb.quat_from_ypr
checks.append(within_tol(rb.quat_from_ypr(0, 0, 0),\
                       np.array([[1],[0],[0],[0]]),tol))
checks.append(within_tol(rb.quat_from_ypr(90, 0, 0),\
                       np.array([[0.52532199], [0.0], [0.0],[0.85090352]]),tol))
checks.append(within_tol(rb.quat_from_ypr(0, 90, 0),\
                       np.array([[0.52532199], [0.0],[0.85090352],[0.0]]),tol))
checks.append(within_tol(rb.quat_from_ypr(0, 0, 90),\
                       np.array([[0.52532199],[0.85090352],[0.0],[0.0]]),tol))
checks.append(within_tol(rb.quat_from_ypr(47,15,6),\
                       np.array([[-0.11087287],[-0.92986012],[0.0086627],[0.35070262]]),tol))
    
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
    
      
    
## Print True if all checks pass
print(all(checks))