import dpctl
import base_kmeans
import numpy
import numba_dppy
from device_selector import get_device_selector

REPEAT = 1
# defines total number of iterations for kmeans accuracy

ITERATIONS = 30

# determine the euclidean distance from the cluster center to each point
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

# assign points to cluster
@numba_dppy.kernel
def calCentroidsSum1(arrayCsum, arrayCnumpoint):
    i = numba_dppy.get_global_id(0)
    arrayCsum[i, 0] = 0
    arrayCsum[i, 1] = 0
    arrayCnumpoint[i] = 0


def calCentroidsSum2(arrayP, arrayPcluster, arrayCsum, arrayCnumpoint, num_points):
    for i in range(num_points):
        ci = arrayPcluster[i]
        arrayCsum[ci, 0] += arrayP[i, 0]
        arrayCsum[ci, 1] += arrayP[i, 1]
        arrayCnumpoint[ci] += 1

# update the centriods array after computation
@numba_dppy.kernel
def updateCentroids(arrayC, arrayCsum, arrayCnumpoint, num_centroids):
    i = numba_dppy.get_global_id(0)
    arrayC[i, 0] = arrayCsum[i, 0] / arrayCnumpoint[i]
    arrayC[i, 1] = arrayCsum[i, 1] / arrayCnumpoint[i]


def kmeans(
    arrayP, arrayPcluster, arrayC, arrayCsum, arrayCnumpoint, num_points, num_centroids
):

    for i in range(ITERATIONS):
        with dpctl.device_context(get_device_selector(is_gpu=True)):
            groupByCluster[num_points, numba_dppy.DEFAULT_LOCAL_SIZE](
                arrayP, arrayPcluster, arrayC, num_points, num_centroids
            )

            calCentroidsSum1[num_centroids, numba_dppy.DEFAULT_LOCAL_SIZE](
                arrayCsum, arrayCnumpoint
            )

        calCentroidsSum2(arrayP, arrayPcluster, arrayCsum, arrayCnumpoint, num_points)

        with dpctl.device_context(get_device_selector(is_gpu=True)):
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

    for i in range(REPEAT):
        for i1 in range(NUMBER_OF_CENTROIDS):
            arrayC[i1, 0] = arrayP[i1, 0]
            arrayC[i1, 1] = arrayP[i1, 1]

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


base_kmeans.run("Kmeans Numba", run_kmeans)
