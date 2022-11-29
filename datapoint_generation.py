from typing import NamedTuple, List, Any, Tuple

import numpy as np
from numpy import ndarray
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


class Cluster(NamedTuple):
    data_points: ndarray
    center: ndarray
    label: int


def generate_clusters(ns_clstr: Any = 2, cluster_std=0.04, n_features: int = 2):
    # centers = [[0.25, 0.25], [0.25, 0.5], [0.5, 0.5]]
    x_f, y_f, centers = make_blobs(n_samples=ns_clstr,
                                   center_box=(0, 1),
                                   cluster_std=cluster_std,
                                   n_features=n_features,
                                   return_centers=True, )

    clstrs: List[Cluster] = []
    for center_idx, center in enumerate(centers):
        clstrs.append(Cluster(x_f[np.isin(y_f, [center_idx])], centers[center_idx], center_idx))

    return clstrs, x_f, y_f


def generate_batch(ns_clstr: Any = 2, cluster_std=0.04, n_features: int = 2):
    x_f, y_f, centers = make_blobs(n_samples=ns_clstr,
                                   center_box=(0, 1),
                                   cluster_std=cluster_std,
                                   n_features=n_features,
                                   return_centers=True, )

    batch: List[Tuple[ndarray, ndarray]] = []
    for label_idx, label in enumerate(y_f):
        datapoint = x_f[label_idx]
        batch.append((datapoint, np.array([0, 1]) if label == 0 else np.array([1, 0])))

    train_batch = batch[:int(len(batch) / 100 * 70)]
    test_batch = batch[int(len(batch) / 100 * 70):]

    return train_batch, test_batch


def plot_blobs(x: ndarray, y: ndarray):
    plt.figure(1)
    plt.clf()
    plt.scatter(x[:, 0], x[:, 1], marker="o", c=y, s=25, edgecolor="k")
    plt.title("Generated clusters")
    plt.show()


if __name__ == '__main__':
    clusters, x, y = generate_clusters(ns_clstr=[2, 2, 1], cluster_std=0.04, n_features=2)
    print(f'{clusters[2].center=}')

    plot_blobs(x, y)
