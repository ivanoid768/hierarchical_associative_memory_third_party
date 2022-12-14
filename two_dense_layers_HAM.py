from dataclasses import dataclass
from typing import NamedTuple, List

import numpy as np
from numpy import ndarray

from scipy.special import softmax

from datapoint_generation import Cluster, generate_clusters


class IterState(NamedTuple):
    x: ndarray
    y: ndarray
    z: ndarray


class IterError(NamedTuple):
    x: ndarray
    y: ndarray
    z: ndarray


@dataclass
class DeltaWeight:
    xy: ndarray
    yz: ndarray


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
W_xy: ndarray = np.random.rand(N_y, N_x) * 0.001
W_yz: ndarray = np.random.rand(N_z, N_y) * 0.001


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
    real_iter_cnt = 0
    for iter_idx in range(iter_cnt):
        prev_x = np.copy(x)
        prev_y = np.copy(y)
        prev_z = np.copy(z)

        x = x_update(prev_x, prev_y, W_xy, t_x)
        y = y_update(prev_x, prev_y, prev_z, W_xy, W_yz, t_y)
        z = z_update(prev_y, W_yz)

        energy = energy_func(x, y, z, W_xy)
        # print(f'{iter_idx=} : {energy=}')
        # print(f'{(energy - prev_energy)=}')

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

        real_iter_cnt += 1

    print(f'{energy=}')
    print(f'{(energy - prev_energy)=}')
    print(f'{real_iter_cnt=}')

    return iters_states


def test_feedforward():
    inp = np.random.rand(x.size)

    iter_outs = feedforward_sync(inp, y, z, W_xy, W_yz, iter_cnt=100)
    print(f'{len(iter_outs)=}, {iter_outs[0].z.shape}')


def x_fd(x: ndarray):
    return np.ones(x.size)


def y_fd(y, err_grad: ndarray):
    sfm = np.reshape(y, (1, -1))
    grad = np.reshape(err_grad, (1, -1))

    d_softmax = (sfm * np.identity(sfm.size) - sfm.transpose() @ sfm)

    return (grad @ d_softmax).ravel()


def z_fd(z: ndarray, err_grad: ndarray):
    sfm = np.reshape(z, (1, -1))
    grad = np.reshape(err_grad, (1, -1))

    d_softmax = (sfm * np.identity(sfm.size) - sfm.transpose() @ sfm)

    return (grad @ d_softmax).ravel()


def train_last_iter(inp: ndarray, iter_state: IterState, lr: float, W_xy: ndarray, W_yz: ndarray):
    (x, y, z) = iter_state

    x_err: ndarray = (x - inp) * x_fd(x)
    xy_dW = np.dot(y[np.newaxis].T, x_err[np.newaxis])

    y_err: ndarray = y_fd(y, np.dot(x_err, W_xy.T))
    # yz_dW = np.dot(y_err, z.T)
    yz_dW = np.dot(z[np.newaxis].T, y_err[np.newaxis])

    z_err: ndarray = z_fd(z, np.dot(y_err, W_yz.T))

    error = IterError(x_err, y_err, z_err)
    dW_sum = DeltaWeight(xy_dW, yz_dW)
    return error, dW_sum


def train_iter(iter_state: IterState, prev_err: IterError,
               W_xy: ndarray, W_yz: ndarray, dW_sum: DeltaWeight, ):
    (x, y, z) = iter_state

    x_err: ndarray = np.dot(prev_err.y, W_xy) * x_fd(x)
    xy_dW = np.dot(x[np.newaxis].T, prev_err.y[np.newaxis])

    y_err_from_x: ndarray = np.dot(prev_err.x, W_xy.T)
    y_err_from_z: ndarray = np.dot(prev_err.z, W_yz)
    y_err: ndarray = y_fd(y, (y_err_from_x + y_err_from_z) / 2)

    xy_dW_from_x = np.dot(y[np.newaxis].T, prev_err.x[np.newaxis])
    yz_dW_from_z = np.dot(y[np.newaxis].T, prev_err.z[np.newaxis])

    z_err: ndarray = z_fd(z, np.dot(prev_err.y, W_yz.T))
    yz_dW = np.dot(z[np.newaxis].T, prev_err.y[np.newaxis])

    dW_sum.xy += (xy_dW.T + xy_dW_from_x) / 2
    dW_sum.yz += (yz_dW + yz_dW_from_z.T) / 2

    return IterError(x_err, y_err, z_err), dW_sum


def update_weights(lr: float, W_xy: ndarray, W_yz: ndarray, dW_sum: DeltaWeight, iter_states_len: int):
    W_xy -= lr * (dW_sum.xy / iter_states_len)
    W_yz -= lr * (dW_sum.yz / iter_states_len)

    return W_xy, W_yz


def test_iter_train(lr0: float = 0.01, iter_cnt: int = 100):
    inp = np.random.rand(x.size)

    iter_states = feedforward_sync(inp, y, z, W_xy, W_yz, iter_cnt=iter_cnt)
    print(f'{len(iter_states)=}, {iter_states[0].z.shape}')
    mse = np.sum((iter_states[-1].x - inp) ** 2) / x.size
    print(f'{mse=}')

    iter_err, dW_sum = train_last_iter(inp, iter_states[-1], lr=lr0, W_xy=W_xy, W_yz=W_yz)
    # dW_sum.yz.fill(0)
    # dW_sum.yz.fill(0)

    prev_mse = mse
    first_mse = mse
    for iter_state in reversed(iter_states[0: -1]):
        iter_err, dW_sum = train_iter(iter_state=iter_state, prev_err=iter_err, W_xy=W_xy, W_yz=W_yz, dW_sum=dW_sum)

    update_weights(lr=lr0, W_xy=W_xy, W_yz=W_yz, dW_sum=dW_sum, iter_states_len=len(iter_states))

    iter_states = feedforward_sync(inp, y, z, W_xy, W_yz, iter_cnt=iter_cnt)
    print(f'{len(iter_states)=}, {iter_states[0].z.shape}')

    mse = np.sum((iter_states[-1].x - inp) ** 2) / x.size
    print(f'{mse=}')
    # mse = np.sum((x - inp) ** 2) / x.size
    print(f'{prev_mse=} {mse=} {(mse - prev_mse)=}')
    print(f'{first_mse=} {mse=} {(mse - first_mse)=}')


def train_one(epoch_cnt: int = 10, lr0: float = 0.01, iter_cnt: int = 100,
              last_iterations_to_train: int = 10, ):
    inp = np.random.rand(x.size)
    iter_states = feedforward_sync(inp, y, z, W_xy, W_yz, iter_cnt=iter_cnt)
    first_mse = np.sum((iter_states[-1].x - inp) ** 2) / x.size

    for epoch_idx in range(epoch_cnt):
        print(f'{epoch_idx=}')
        lr = (epoch_cnt - epoch_idx) * lr0
        print(f'{lr=}')

        iter_states = feedforward_sync(inp, y, z, W_xy, W_yz, iter_cnt=iter_cnt)
        iter_err, dW_sum = train_last_iter(inp, iter_states[-1], lr=lr, W_xy=W_xy, W_yz=W_yz)
        # dW_sum.yz.fill(0)
        # dW_sum.yz.fill(0)
        for iter_state in reversed(iter_states[-(last_iterations_to_train + 1): -1]):
            iter_err, dW_sum = train_iter(iter_state=iter_state, prev_err=iter_err, W_xy=W_xy, W_yz=W_yz, dW_sum=dW_sum)
        update_weights(lr=lr, W_xy=W_xy, W_yz=W_yz, dW_sum=dW_sum, iter_states_len=(len(iter_states) - 1))

    iter_states = feedforward_sync(inp, y, z, W_xy, W_yz, iter_cnt=iter_cnt)
    mse = np.sum((iter_states[-1].x - inp) ** 2) / x.size
    print(f'{first_mse=} {mse=} {(mse - first_mse)=}')


def train(inp_batch: List[ndarray],
          epoch_cnt: int = 10,
          lr0: float = 0.01,
          iter_cnt: int = 100,
          last_iterations_to_train: int = 10, ):
    global W_xy, W_yz

    mse_sum = 0
    for inp in inp_batch:
        iter_states = feedforward_sync(inp, y, z, W_xy, W_yz, iter_cnt=iter_cnt)
        mse_sum += np.sum((iter_states[-1].x - inp) ** 2) / x.size
    first_mse = mse_sum / len(inp_batch)

    for epoch_idx in range(epoch_cnt):
        print(f'{epoch_idx=}')
        lr = (epoch_cnt - epoch_idx) * lr0
        print(f'{lr=}')

        dW_batch = DeltaWeight(np.zeros(W_xy.shape), np.zeros(W_yz.shape))
        for inp in inp_batch:
            iter_states = feedforward_sync(inp, y, z, W_xy, W_yz, iter_cnt=iter_cnt)
            iter_err, dW_sum = train_last_iter(inp, iter_states[-1], lr=lr, W_xy=W_xy, W_yz=W_yz)
            for iter_state in reversed(iter_states[-(last_iterations_to_train + 1): -1]):
                iter_err, dW_sum = train_iter(iter_state=iter_state, prev_err=iter_err, W_xy=W_xy, W_yz=W_yz,
                                              dW_sum=dW_sum)

            trained_iter_cnt = min(last_iterations_to_train, len(iter_states))
            trained_iter_cnt = trained_iter_cnt - 1 if trained_iter_cnt > 1 else 1
            dW_batch.xy += (dW_sum.xy / trained_iter_cnt)
            dW_batch.yz += (dW_sum.yz / trained_iter_cnt)

        dW_batch.xy = dW_batch.xy / len(inp_batch)
        dW_batch.yz = dW_batch.yz / len(inp_batch)
        W_xy -= lr * dW_batch.xy
        W_yz -= lr * dW_batch.yz

    mse = 0
    for inp in inp_batch:
        iter_states = feedforward_sync(inp, y, z, W_xy, W_yz, iter_cnt=iter_cnt)
        mse += np.sum((iter_states[-1].x - inp) ** 2) / x.size
    mse = mse / len(inp_batch)

    print(f'{first_mse=} {mse=} {(mse - first_mse)=} {(first_mse / mse if mse else 0)=}')


def test_train():
    global x

    clusters, x, _ = generate_clusters(ns_clstr=[2, 2, 1], cluster_std=0.04, n_features=x.size)

    batch = [clusters[0].data_points[0], clusters[1].data_points[0]]
    train(inp_batch=batch, epoch_cnt=100, lr0=0.07, iter_cnt=100, last_iterations_to_train=10)

    real_iter_cnt = []

    print(f'cluster_datapoint')
    trained_cluster_datapoint = clusters[0].data_points[0]
    iter_states = feedforward_sync(trained_cluster_datapoint, y, z, W_xy, W_yz, iter_cnt=100)
    real_iter_cnt.append(len(iter_states))

    print(f'trained_cluster_datapoint')
    trained_cluster_datapoint = clusters[0].data_points[1]
    iter_states = feedforward_sync(trained_cluster_datapoint, y, z, W_xy, W_yz, iter_cnt=100)
    real_iter_cnt.append(len(iter_states))

    print(f'trained_cluster_datapoint_2')
    trained_cluster_datapoint = clusters[1].data_points[1]
    iter_states = feedforward_sync(trained_cluster_datapoint, y, z, W_xy, W_yz, iter_cnt=100)
    real_iter_cnt.append(len(iter_states))

    print(f'new_cluster_datapoint')
    trained_cluster_datapoint = clusters[2].data_points[0]
    iter_states = feedforward_sync(trained_cluster_datapoint, y, z, W_xy, W_yz, iter_cnt=100)
    real_iter_cnt.append(len(iter_states))

    print(f'{real_iter_cnt=}')


# test_feedforward()
# test_iter_train(lr0=0.01)
# train_one(epoch_cnt=100, lr0=0.1, iter_cnt=100, last_iterations_to_train=2)
test_train()
