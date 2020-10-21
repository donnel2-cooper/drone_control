# Code for pole plots
import mavsim_python_chap5_model_coef as uav
import numpy as np
import control.matlab as ctrl
import matplotlib.pyplot as plt

poles_lon, modes_lon = np.linalg.eig(uav.A_lon)
poles_lat, modes_lat = np.linalg.eig(uav.A_lat)
ii = 0

# print(np.linalg.matrix_rank(uav.A_lon))
# print(np.linalg.matrix_rank(uav.A_lat))

for x in range(len(modes_lon)):
    plt.figure()
    mode = modes_lon[:,x]
    for y in range(len(mode)):
        plt.polar([0,np.angle(mode[y])],[0,abs(mode[y])],marker='o')
    plt.legend(['u','alpha/w','q','theta','h'])

for x in range(len(modes_lat)):
    plt.figure()
    mode = modes_lat[:,x]
    for y in range(len(mode)):
        plt.polar([0,np.angle(mode[y])],[0,abs(mode[y])],marker='o')
    plt.legend(['beta/v','p','r','phi','psi'])

plt.figure()
plt.scatter([temp.real for temp in poles_lon],[temp.imag for temp in poles_lon],marker = 'x')
plt.title("Longitudinal Poles")

plt.figure()
plt.scatter([temp.real for temp in poles_lat],[temp.imag for temp in poles_lat],marker = 'x')
plt.title("Lateral Poles")