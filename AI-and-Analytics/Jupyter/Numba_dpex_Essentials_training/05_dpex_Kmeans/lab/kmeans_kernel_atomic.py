import base_kmeans_gpu

from math import sqrt

import numba_dpex as nbdx


@nbdx.kernel
def groupByCluster(arrayP, arrayPcluster, arrayC, num_points, num_centroids):
    idx = nbdx.get_global_id(0)
    # if idx < num_points: # why it was removed??
    minor_distance = -1
    for i in range(num_centroids):
        dx = arrayP[idx, 0] - arrayC[i, 0]
        dy = arrayP[idx, 1] - arrayC[i, 1]
        my_distance = sqrt(dx * dx + dy * dy)
        if minor_distance > my_distance or minor_distance == -1:
            minor_distance = my_distance
            arrayPcluster[idx] = i


@nbdx.kernel
def calCentroidsSum1(arrayCsum, arrayCnumpoint):
    i = nbdx.get_global_id(0)
    arrayCsum[i, 0] = 0
    arrayCsum[i, 1] = 0
    arrayCnumpoint[i] = 0


@nbdx.kernel
def calCentroidsSum2(arrayP, arrayPcluster, arrayCsum, arrayCnumpoint):
    i = nbdx.get_global_id(0)
    ci = arrayPcluster[i]
    nbdx.atomic.add(arrayCsum, (ci, 0), arrayP[i, 0])
    nbdx.atomic.add(arrayCsum, (ci, 1), arrayP[i, 1])
    nbdx.atomic.add(arrayCnumpoint, ci, 1)


@nbdx.kernel
def updateCentroids(arrayC, arrayCsum, arrayCnumpoint, num_centroids):
    i = nbdx.get_global_id(0)
    arrayC[i, 0] = arrayCsum[i, 0] / arrayCnumpoint[i]
    arrayC[i, 1] = arrayCsum[i, 1] / arrayCnumpoint[i]


@nbdx.kernel
def copy_arrayC(arrayC, arrayP):
    i = nbdx.get_global_id(0)
    arrayC[i, 0] = arrayP[i, 0]
    arrayC[i, 1] = arrayP[i, 1]


def kmeans_kernel(
    arrayP,
    arrayPcluster,
    arrayC,
    arrayCsum,
    arrayCnumpoint,
    niters,
    num_points,
    num_centroids,
):
    copy_arrayC[num_centroids,](arrayC, arrayP)

    for i in range(niters):
        groupByCluster[num_points,](
            arrayP, arrayPcluster, arrayC, num_points, num_centroids
        )

        calCentroidsSum1[num_centroids,](arrayCsum, arrayCnumpoint)

        calCentroidsSum2[num_points,](
            arrayP, arrayPcluster, arrayCsum, arrayCnumpoint
        )

        updateCentroids[num_centroids,](
            arrayC, arrayCsum, arrayCnumpoint, num_centroids
        )

    return arrayC, arrayCsum, arrayCnumpoint


def kmeans(
    arrayP,
    arrayPclusters,
    arrayC,
    arrayCsum,
    arrayCnumpoint,
    niters,
    npoints,
    ndims,
    ncentroids,
):
    arrayC, arrayCsum, arrayCnumpoint = kmeans_kernel(
        arrayP,
        arrayPclusters,
        arrayC,
        arrayCsum,
        arrayCnumpoint,
        niters,
        npoints,        
        ncentroids,
    )

base_kmeans_gpu.run("Kmeans Numba", kmeans)
