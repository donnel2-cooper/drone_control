import sys
sys.path.append('..')
import numpy as np
from tools.rotations import Euler2Quaternion
import parameters.aerosonde_parameters as MAV
import aerosonde_parameters as MAV
from tools.rotations import Quaternion2Rotation, Quaternion2Euler
import mav_dynamics
import mavsim_python_chap5_model_coef as chap5

# Define sim parameters
x0 = chap5.x_trim