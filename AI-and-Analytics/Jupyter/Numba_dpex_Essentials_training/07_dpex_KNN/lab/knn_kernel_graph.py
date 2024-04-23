import dpctl
import math
import dpnp as np

import base_knn_graph
from device_selector import get_device_selector

from numba_dpex import kernel, DEFAULT_LOCAL_SIZE
import numba_dpex

@kernel(
    access_types={
        "read_only": ["train", "train_labels", "test", "votes_to_classes_lst"],
        "write_only": ["predictions"],
    }
)
def run_knn_kernel(
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
    i = numba_dpex.get_global_id(0)
    queue_neighbors = numba_dpex.private.array(shape=(5, 2), dtype=np.float64)

    for j in range(k):
        x1 = train[j]
        x2 = test[i]

        distance = 0.0
        for jj in range(data_dim):
            diff = x1[jj] - x2[jj]
            distance += diff * diff
        dist = math.sqrt(distance)

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

        distance = 0.0
        for jj in range(data_dim):
            diff = x1[jj] - x2[jj]
            distance += diff * diff
        dist = math.sqrt(distance)

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

    for j in range(k):
        votes_to_classes[int(queue_neighbors[j, 1])] += 1

    max_ind = 0
    max_value = 0

    for j in range(classes_num):
        if votes_to_classes[j] > max_value:
            max_value = votes_to_classes[j]
            max_ind = j

    predictions[i] = max_ind


def run_knn(
    train,
    train_labels,
    test,
    k,
    classes_num,
    test_size,
    train_size,
    predictions,
    votes_to_classes_lst,
    data_dim,
):
    with dpctl.device_context(get_device_selector(is_gpu=True)) as gpu_queue:
        run_knn_kernel[test_size, numba_dpex.DEFAULT_LOCAL_SIZE](
            train,
            train_labels,
            test,
            k,
            classes_num,
            train_size,
            predictions,
            votes_to_classes_lst,
            data_dim,
        )


base_knn_graph.run("K-Nearest-Neighbors Numba", run_knn)
