import sys
sys.path.append('..')
import numpy as np
import parameters.aerosonde_parameters as MAV
from tools.rotations import Quaternion2Rotation, Quaternion2Euler, Euler2Quaternion
import mav_dynamics
import mavsim_python_chap5_model_coef as chap5

# Define sim parameters
x0 = chap5.x_trim