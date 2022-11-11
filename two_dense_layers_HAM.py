import numpy as np
from numpy import ndarray

from scipy.special import softmax

N_x = 128
N_y = 128
N_z = 128

b2 = 0.2
b3 = 0.3

t_x = 0.2
t_y = 0.02

# 3 layers of neurons - output arrays
x = np.zeros(N_x)
y = np.zeros(N_y)
z = np.zeros(N_z)

# synapse weights matrices
W_xy = np.random.rand(N_x, N_y) * 0.001
W_yz = np.random.rand(N_y, N_z) * 0.001


def z_update(y: ndarray, W_yz: ndarray):
    return np.dot(W_yz, softmax(b2 * y))


def y_update(x: ndarray, y: ndarray, z: ndarray, W_xy: ndarray, W_yz: ndarray, t: float):
    W_yzT = W_yz.T

    return t * (np.dot(W_yzT, softmax(b3 * z)) + np.dot(W_xy, x) - y) + y


def x_update(x: ndarray, y: ndarray, W_xy: ndarray, t: float):
    W_xyT = W_xy.T

    return t * (np.dot(W_xyT, softmax(b2 * y)) - x) + x


def energy_last_term(x: ndarray, y: ndarray, W_xy: ndarray):
    y_sf = softmax(b2 * y)
    x_out = np.dot(W_xy, x)

    return np.sum(y_sf * x_out[np.newaxis].T)


def energy_func(x: ndarray, y: ndarray, z: ndarray, W_xy: ndarray):
    energy = np.sum(x ** 2) / 2
    energy += np.sum(softmax(b2 * y) * y)
    energy -= np.log(np.sum(np.exp(b2 * y))) / b2
    energy -= np.log(np.sum(np.exp(b3 * z))) / b3
    energy -= energy_last_term(x, y, W_xy)

    return energy


def feedforward_sync(inp: ndarray, y: ndarray, z: ndarray, W_xy: ndarray, W_yz: ndarray, iter_cnt: int = 100):
    x = np.copy(inp)

    energy = energy_func(x, y, z, W_xy)
    print(f'start_{energy=}')

    prev_energy = energy
    for iter_idx in range(iter_cnt):
        prev_x = np.copy(x)
        prev_y = np.copy(y)
        prev_z = np.copy(z)

        x = x_update(prev_x, prev_y, W_xy, t_x)
        y = y_update(prev_x, prev_y, prev_z, W_xy, W_yz, t_y)
        z = z_update(prev_y, W_yz)

        energy = energy_func(x, y, z, W_xy)
        print(f'{iter_idx=} : {energy=}')
        print(f'{(energy - prev_energy)=}')
        if (energy - prev_energy) == 0.0:
            break

        prev_energy = energy


def test_feedforward():
    inp = np.random.rand(x.size)

    feedforward_sync(inp, y, z, W_xy, W_yz, iter_cnt=100 * 5)


test_feedforward()
