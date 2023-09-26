# *****************************************************************************
# Copyright (c) 2020, Intel Corporation All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#     Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# *****************************************************************************

import base_knn
import dpnp as np

import numba

DATA_DIM = 16

# Define the number of nearest neighbors
k = 5

# @numba.jit(nopython=True)
# def euclidean_dist(x1, x2):
#     return np.linalg.norm(x1-x2)


@numba.jit(nopython=True)
def euclidean_dist(x1, x2):
    distance = 0

    for i in range(DATA_DIM):
        diff = x1[i] - x2[i]
        distance += diff * diff

    result = distance**0.5
    # result = np.sqrt(distance)
    return result


@numba.jit(nopython=True)
def push_queue(queue_neighbors, new_distance, index=4):
    while index > 0 and new_distance[0] < queue_neighbors[index - 1, 0]:
        queue_neighbors[index] = queue_neighbors[index - 1]
        index = index - 1
        queue_neighbors[index] = new_distance


@numba.jit(nopython=True)
def sort_queue(queue_neighbors):
    for i in range(len(queue_neighbors)):
        push_queue(queue_neighbors, queue_neighbors[i], i)


@numba.jit(nopython=True)
def simple_vote(neighbors, classes_num):
    votes_to_classes = np.zeros(classes_num)

    for i in range(len(neighbors)):
        votes_to_classes[int(neighbors[i, 1])] += 1

    max_ind = 0
    max_value = 0

    for i in range(classes_num):
        if votes_to_classes[i] > max_value:
            max_value = votes_to_classes[i]
            max_ind = i

    return max_ind


@numba.jit(nopython=True, parallel=True)
def run_knn_kernel(train, train_labels, test, k, classes_num):
    test_size = len(test)
    train_size = len(train)

    predictions = np.empty(test_size)
    queue_neighbors_lst = np.empty((test_size, k, 2))

    for i in numba.prange(test_size):
        queue_neighbors = queue_neighbors_lst[i]

        for j in range(k):
            dist = euclidean_dist(train[j], test[i])
            # queue_neighbors[j] = (dist, train_labels[j])
            queue_neighbors[j, 0] = dist
            queue_neighbors[j, 1] = train_labels[j]
            # queue_neighbors.append((dist, train_labels[j]))

        sort_queue(queue_neighbors)

        for j in range(k, train_size):
            dist = euclidean_dist(train[j], test[i])

            if dist < queue_neighbors[k - 1][0]:
                # queue_neighbors[k - 1] = new_neighbor
                queue_neighbors[k - 1][0] = dist
                queue_neighbors[k - 1][1] = train_labels[j]
                push_queue(queue_neighbors, queue_neighbors[k - 1])

        predictions[i] = simple_vote(queue_neighbors, classes_num)

    return predictions


def run_knn(train, train_labels, test, k=5, classes_num=3):
    run_knn_kernel(train, train_labels, test, k, classes_num)


base_knn.run("K-Nearest-Neighbors Numba", run_knn)
