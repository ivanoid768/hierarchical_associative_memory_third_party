import numpy as np

N_x = 128
N_y = 128
N_z = 128

# 3 layers of neurons - output arrays
x = np.zeros(N_x)
y = np.zeros(N_y)
z = np.zeros(N_z)

# synapse weights matrices
W_xy = np.random.rand(N_x, N_y)
W_yz = np.random.rand(N_y, N_z)

