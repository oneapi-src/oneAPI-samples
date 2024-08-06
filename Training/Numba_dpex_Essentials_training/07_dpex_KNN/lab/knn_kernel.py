# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from math import sqrt
import base_knn

import numba_dpex as dpex
import numpy as np


@dpex.kernel
def _knn_kernel(  # noqa: C901: TODO: can we simplify logic?
    train,
    train_labels,
    test,
    k,
    classes_num,
    train_size,
    predictions,
    votes_to_classes_lst,
    data_dim,
):
    dtype = train.dtype
    i = dpex.get_global_id(0)
    # here k has to be 5 in order to match with numpy
    queue_neighbors = dpex.private.array(shape=(5, 2), dtype=dtype)

    for j in range(k):
        x1 = train[j]
        x2 = test[i]

        distance = dtype.type(0.0)
        for jj in range(data_dim):
            diff = x1[jj] - x2[jj]
            distance += diff * diff
        dist = sqrt(distance)

        queue_neighbors[j, 0] = dist
        queue_neighbors[j, 1] = train_labels[j]

    for j in range(k):
        new_distance = queue_neighbors[j, 0]
        new_neighbor_label = queue_neighbors[j, 1]
        index = j

        while index > 0 and new_distance < queue_neighbors[index - 1, 0]:
            queue_neighbors[index, 0] = queue_neighbors[index - 1, 0]
            queue_neighbors[index, 1] = queue_neighbors[index - 1, 1]

            index = index - 1

            queue_neighbors[index, 0] = new_distance
            queue_neighbors[index, 1] = new_neighbor_label

    for j in range(k, train_size):
        x1 = train[j]
        x2 = test[i]

        distance = dtype.type(0.0)
        for jj in range(data_dim):
            diff = x1[jj] - x2[jj]
            distance += diff * diff
        dist = sqrt(distance)

        if dist < queue_neighbors[k - 1][0]:
            queue_neighbors[k - 1][0] = dist
            queue_neighbors[k - 1][1] = train_labels[j]
            new_distance = queue_neighbors[k - 1, 0]
            new_neighbor_label = queue_neighbors[k - 1, 1]
            index = k - 1

            while index > 0 and new_distance < queue_neighbors[index - 1, 0]:
                queue_neighbors[index, 0] = queue_neighbors[index - 1, 0]
                queue_neighbors[index, 1] = queue_neighbors[index - 1, 1]

                index = index - 1

                queue_neighbors[index, 0] = new_distance
                queue_neighbors[index, 1] = new_neighbor_label

    votes_to_classes = votes_to_classes_lst[i]

    for j in range(len(queue_neighbors)):
        votes_to_classes[int(queue_neighbors[j, 1])] += 1

    max_ind = 0
    max_value = dtype.type(0)

    for j in range(classes_num):
        if votes_to_classes[j] > max_value:
            max_value = votes_to_classes[j]
            max_ind = j

    predictions[i] = max_ind


def knn(
    x_train,
    y_train,
    x_test,
    k,
    classes_num,
    test_size,
    train_size,
    predictions,
    votes_to_classes,
    data_dim,
):
    _knn_kernel[test_size,](
        x_train,
        y_train,
        x_test,
        k,
        classes_num,
        train_size,
        predictions,
        votes_to_classes,
        data_dim,
    )
