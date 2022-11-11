import numpy as np
from numpy import ndarray

from scipy.special import softmax

N_x = 128
N_y = 128
N_z = 128

b2 = 0.2
b3 = 0.3

# 3 layers of neurons - output arrays
x = np.zeros(N_x)
y = np.zeros(N_y)
z = np.zeros(N_z)

# synapse weights matrices
W_xy = np.random.rand(N_x, N_y)
W_yz = np.random.rand(N_y, N_z)


def z_update(y: ndarray, z: ndarray, W_yz: ndarray):
    z += np.dot(W_yz, softmax(b2 * y)) - z


def y_update(x: ndarray, y: ndarray, z: ndarray, W_xy: ndarray, W_yz: ndarray):
    W_yzT = W_yz.T

    y += np.dot(W_yzT, softmax(b3 * z)) + np.dot(W_xy, x) - y


def x_update(x: ndarray, y: ndarray, W_xy: ndarray):
    W_xyT = W_xy.T

    x += np.dot(W_xyT, softmax(b2 * y)) - x


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


def feedforward():
    pass


def test_feedforward():
    pass
