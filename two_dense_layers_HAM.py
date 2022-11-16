from collections import namedtuple
from typing import NamedTuple, List

import numpy as np
from numpy import ndarray

from scipy.special import softmax


class IterState(NamedTuple):
    x: ndarray
    y: ndarray
    z: ndarray


N_x = 128
N_y = 128 * 2
N_z = 128

b2 = 0.2
b3 = 0.3

speed = 0.1

t_x = 0.2 * speed
t_y = 0.002 * speed

# 3 layers of neurons - output arrays
x = np.zeros(N_x)
y = np.zeros(N_y)
z = np.zeros(N_z)

# synapse weights matrices
W_xy = np.random.rand(N_y, N_x) * 0.001
W_yz = np.random.rand(N_z, N_y) * 0.001


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

    iters_states = []

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

        if (energy - prev_energy) >= 0.0:
            x = np.copy(prev_x)
            y = np.copy(prev_y)
            z = np.copy(prev_z)

            iter_state = IterState(x, y, z)
            iters_states.append(iter_state)
            break

        prev_energy = energy

        iter_state = IterState(x, y, z)
        iters_states.append(iter_state)

    return iters_states


def test_feedforward():
    inp = np.random.rand(x.size)

    iter_outs = feedforward_sync(inp, y, z, W_xy, W_yz, iter_cnt=100)
    print(f'{len(iter_outs)=}, {iter_outs[0].z.shape}')


def x_fd(x: ndarray):
    pass


def y_fd(y, err: ndarray):
    sfm = np.reshape(y, (1, -1))
    grad = np.reshape(err, (1, -1))

    d_softmax = (sfm * np.identity(sfm.size) - sfm.transpose() @ sfm)

    return (grad @ d_softmax).ravel()


def z_fd(err: ndarray):
    s = softmax(err)
    return np.diag(s) - np.outer(s, s)


def train_last_iter(inp: ndarray, iter_state: IterState, lr: float, W_xy: ndarray, W_yz: ndarray):
    x = iter_state.x
    x_err: ndarray = x - inp
    W_xy_update: ndarray = W_xy.T * lr * x_err
    y_bp_out: ndarray = np.dot(W_xy.T, x_err)

    y = iter_state.y
    y_err: ndarray = y - y_bp_out
    W_yz_update: ndarray = W_yz.T * lr * y_fd(y, y_err)
    z_bp_out: ndarray = np.dot(W_yz.T, y_fd(y, y_err))

    z = iter_state.z
    z_err: ndarray = z - z_bp_out

    return x_err, y_err, z_err, W_xy_update, W_yz_update


def train_batch(input_list: List[ndarray], y: ndarray, z: ndarray, W_xy: ndarray, W_yz: ndarray,
                iter_cnt: int = 100,
                epoch_cnt: int = 100, ):
    for inp in input_list:
        iter_outs = feedforward_sync(inp, y, z, W_xy, W_yz, iter_cnt=iter_cnt)

        last_inter_state = iter_outs[-1]

        for iter_idx, iter_state in enumerate(reversed(iter_outs)):
            pass


test_feedforward()
