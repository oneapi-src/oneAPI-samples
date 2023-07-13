import dpctl
import base_kmeans_gpu
import numpy
import numba_dpex as nb
from numba_dpex import atomic

REPEAT = 1
ITERATIONS = 30

atomic_add = atomic.add


@nb.kernel
def groupByCluster(arrayP, arrayPcluster, arrayC, num_points, num_centroids):
    idx = nb.get_global_id(0)
    # if idx < num_points: # why it was removed??
    minor_distance = -1
    for i in range(num_centroids):
        dx = arrayP[idx, 0] - arrayC[i, 0]
        dy = arrayP[idx, 1] - arrayC[i, 1]
        my_distance = numpy.sqrt(dx * dx + dy * dy)
        if minor_distance > my_distance or minor_distance == -1:
            minor_distance = my_distance
            arrayPcluster[idx] = i


@nb.kernel
def calCentroidsSum1(arrayCsum, arrayCnumpoint):
    i = nb.get_global_id(0)
    arrayCsum[i, 0] = 0
    arrayCsum[i, 1] = 0
    arrayCnumpoint[i] = 0


@nb.kernel
def calCentroidsSum2(arrayP, arrayPcluster, arrayCsum, arrayCnumpoint):
    i = nb.get_global_id(0)
    ci = arrayPcluster[i]
    atomic_add(arrayCsum, (ci, 0), arrayP[i, 0])
    atomic_add(arrayCsum, (ci, 1), arrayP[i, 1])
    atomic_add(arrayCnumpoint, ci, 1)


@nb.kernel
def updateCentroids(arrayC, arrayCsum, arrayCnumpoint, num_centroids):
    i = nb.get_global_id(0)
    arrayC[i, 0] = arrayCsum[i, 0] / arrayCnumpoint[i]
    arrayC[i, 1] = arrayCsum[i, 1] / arrayCnumpoint[i]


@nb.kernel
def copy_arrayC(arrayC, arrayP):
    i = nb.get_global_id(0)
    arrayC[i, 0] = arrayP[i, 0]
    arrayC[i, 1] = arrayP[i, 1]


def kmeans(
    arrayP, arrayPcluster, arrayC, arrayCsum, arrayCnumpoint, num_points, num_centroids
):

    copy_arrayC[num_centroids,](arrayC, arrayP)

    for i in range(ITERATIONS):
        groupByCluster[num_points,](
            arrayP, arrayPcluster, arrayC, num_points, num_centroids
        )

        calCentroidsSum1[num_centroids,](
            arrayCsum, arrayCnumpoint
        )

        calCentroidsSum2[num_points,](
            arrayP, arrayPcluster, arrayCsum, arrayCnumpoint
        )

        updateCentroids[num_centroids,](
            arrayC, arrayCsum, arrayCnumpoint, num_centroids
        )

    return arrayC, arrayCsum, arrayCnumpoint


def run_kmeans(
    arrayP,
    arrayPclusters,
    arrayC,
    arrayCsum,
    arrayCnumpoint,
    NUMBER_OF_POINTS,
    NUMBER_OF_CENTROIDS,
):

    with dpctl.device_context(base_kmeans_gpu.get_device_selector(is_gpu=True)):
        for i in range(REPEAT):
            arrayC, arrayCsum, arrayCnumpoint = kmeans(
                arrayP,
                arrayPclusters,
                arrayC,
                arrayCsum,
                arrayCnumpoint,
                NUMBER_OF_POINTS,
                NUMBER_OF_CENTROIDS,
            )


base_kmeans_gpu.run("Kmeans Numba", run_kmeans)
