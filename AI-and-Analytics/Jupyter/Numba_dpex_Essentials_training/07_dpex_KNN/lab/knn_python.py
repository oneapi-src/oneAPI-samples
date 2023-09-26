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

import dpnp as np


def knn_python(
    train,
    train_labels,
    test,
    k,
    classes_num,
    train_size,
    test_size,
    predictions,
    votes_to_classes_lst,
    data_dim,
):
    for i in range(test_size):
        queue_neighbors = np.empty((k, 2))  # queue_neighbors_lst[i]

        for j in range(k):
            # dist = euclidean_dist(train[j], test[i])
            x1 = train[j]
            x2 = test[i]

            distance = 0.0
            for jj in range(data_dim):
                diff = x1[jj] - x2[jj]
                distance += diff * diff
            dist = distance**0.5

            queue_neighbors[j, 0] = dist
            queue_neighbors[j, 1] = train_labels[j]

        # sort_queue(queue_neighbors)
        for j in range(len(queue_neighbors)):
            # push_queue(queue_neighbors, queue_neighbors[i], i)
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
            # dist = euclidean_dist(train[j], test[i])
            x1 = train[j]
            x2 = test[i]

            distance = 0.0
            for jj in range(data_dim):
                diff = x1[jj] - x2[jj]
                distance += diff * diff
            dist = distance**0.5

            if dist < queue_neighbors[k - 1][0]:
                # queue_neighbors[k - 1] = new_neighbor
                queue_neighbors[k - 1][0] = dist
                queue_neighbors[k - 1][1] = train_labels[j]
                # push_queue(queue_neighbors, queue_neighbors[k - 1])
                new_distance = queue_neighbors[k - 1, 0]
                new_neighbor_label = queue_neighbors[k - 1, 1]
                index = k - 1

                while (
                    index > 0 and new_distance < queue_neighbors[index - 1, 0]
                ):
                    queue_neighbors[index, 0] = queue_neighbors[index - 1, 0]
                    queue_neighbors[index, 1] = queue_neighbors[index - 1, 1]

                    index = index - 1

                    queue_neighbors[index, 0] = new_distance
                    queue_neighbors[index, 1] = new_neighbor_label

        votes_to_classes = votes_to_classes_lst[i]

        for j in range(len(queue_neighbors)):
            votes_to_classes[int(queue_neighbors[j, 1])] += 1

        max_ind = 0
        max_value = 0

        for j in range(classes_num):
            if votes_to_classes[j] > max_value:
                max_value = votes_to_classes[j]
                max_ind = j

        predictions[i] = max_ind