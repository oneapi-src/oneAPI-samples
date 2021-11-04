import dpctl
import base_kmeans_gpu
import numpy
import numba_dppy

REPEAT = 1

ITERATIONS = 30


@numba_dppy.kernel
def groupByCluster(arrayP, arrayPcluster, arrayC, num_points, num_centroids):
    idx = numba_dppy.get_global_id(0)
    if idx < num_points:
        minor_distance = -1
        for i in range(num_centroids):
            dx = arrayP[idx, 0] - arrayC[i, 0]
            dy = arrayP[idx, 1] - arrayC[i, 1]
            my_distance = numpy.sqrt(dx * dx + dy * dy)
            if minor_distance > my_distance or minor_distance == -1:
                minor_distance = my_distance
                arrayPcluster[idx] = i


@numba_dppy.kernel
def calCentroidsSum1(arrayCsum, arrayCnumpoint):
    i = numba_dppy.get_global_id(0)
    arrayCsum[i, 0] = 0
    arrayCsum[i, 1] = 0
    arrayCnumpoint[i] = 0


@numba_dppy.kernel
def calCentroidsSum2(arrayP, arrayPcluster, arrayCsum, arrayCnumpoint):
    i = numba_dppy.get_global_id(0)
    ci = arrayPcluster[i]
    numba_dppy.atomic.add(arrayCsum, (ci, 0), arrayP[i, 0])
    numba_dppy.atomic.add(arrayCsum, (ci, 1), arrayP[i, 1])
    numba_dppy.atomic.add(arrayCnumpoint, ci, 1)


@numba_dppy.kernel
def updateCentroids(arrayC, arrayCsum, arrayCnumpoint, num_centroids):
    i = numba_dppy.get_global_id(0)
    arrayC[i, 0] = arrayCsum[i, 0] / arrayCnumpoint[i]
    arrayC[i, 1] = arrayCsum[i, 1] / arrayCnumpoint[i]


@numba_dppy.kernel
def copy_arrayC(arrayC, arrayP):
    i = numba_dppy.get_global_id(0)
    arrayC[i, 0] = arrayP[i, 0]
    arrayC[i, 1] = arrayP[i, 1]


def kmeans(
    arrayP, arrayPcluster, arrayC, arrayCsum, arrayCnumpoint, num_points, num_centroids
):

    copy_arrayC[num_centroids, numba_dppy.DEFAULT_LOCAL_SIZE](arrayC, arrayP)

    for i in range(ITERATIONS):
        groupByCluster[num_points, numba_dppy.DEFAULT_LOCAL_SIZE](
            arrayP, arrayPcluster, arrayC, num_points, num_centroids
        )

        calCentroidsSum1[num_centroids, numba_dppy.DEFAULT_LOCAL_SIZE](
            arrayCsum,
            arrayCnumpoint,
        )

        calCentroidsSum2[num_points, numba_dppy.DEFAULT_LOCAL_SIZE](
            arrayP,
            arrayPcluster,
            arrayCsum,
            arrayCnumpoint,
        )

        updateCentroids[num_centroids, numba_dppy.DEFAULT_LOCAL_SIZE](
            arrayC, arrayCsum, arrayCnumpoint, num_centroids
        )

    return arrayC, arrayCsum, arrayCnumpoint


def printCentroid(arrayC, arrayCsum, arrayCnumpoint, NUMBER_OF_CENTROIDS):
    for i in range(NUMBER_OF_CENTROIDS):
        print(
            "[x={:6f}, y={:6f}, x_sum={:6f}, y_sum={:6f}, num_points={:d}]".format(
                arrayC[i, 0],
                arrayC[i, 1],
                arrayCsum[i, 0],
                arrayCsum[i, 1],
                arrayCnumpoint[i],
            )
        )

    print("--------------------------------------------------")


def run_kmeans(
    arrayP,
    arrayPclusters,
    arrayC,
    arrayCsum,
    arrayCnumpoint,
    NUMBER_OF_POINTS,
    NUMBER_OF_CENTROIDS,
):

    with dpctl.device_context(base_kmeans_gpu.get_device_selector()):
        for i in range(REPEAT):
            # for i1 in range(NUMBER_OF_CENTROIDS):
            #     arrayC[i1, 0] = arrayP[i1, 0]
            #     arrayC[i1, 1] = arrayP[i1, 1]

            arrayC, arrayCsum, arrayCnumpoint = kmeans(
                arrayP,
                arrayPclusters,
                arrayC,
                arrayCsum,
                arrayCnumpoint,
                NUMBER_OF_POINTS,
                NUMBER_OF_CENTROIDS,
            )

    #     if i + 1 == REPEAT:
    #         printCentroid(arrayC, arrayCsum, arrayCnumpoint, NUMBER_OF_CENTROIDS)

    # print("Iterations: {:d}".format(ITERATIONS))


base_kmeans_gpu.run("Kmeans Numba", run_kmeans)
