
import dpctl
import base_kmeans_gpu
import numpy
import numba

REPEAT = 1

# defines total number of iterations for kmeans accuracy
ITERATIONS = 30

# determine the euclidean distance from the cluster center to each point
@numba.jit(nopython=True, parallel=True, fastmath=True)
def groupByCluster(arrayP, arrayPcluster, arrayC, num_points, num_centroids):
    # parallel for loop
    for i0 in numba.prange(num_points):
        minor_distance = -1
        for i1 in range(num_centroids):
            dx = arrayP[i0, 0] - arrayC[i1, 0]
            dy = arrayP[i0, 1] - arrayC[i1, 1]
            my_distance = numpy.sqrt(dx * dx + dy * dy)
            if minor_distance > my_distance or minor_distance == -1:
                minor_distance = my_distance
                arrayPcluster[i0] = i1
    return arrayPcluster


# assign points to cluster
@numba.jit(nopython=True, parallel=True, fastmath=True)
def calCentroidsSum(
    arrayP, arrayPcluster, arrayCsum, arrayCnumpoint, num_points, num_centroids
):
    # parallel for loop
    for i in numba.prange(num_centroids):
        arrayCsum[i, 0] = 0
        arrayCsum[i, 1] = 0
        arrayCnumpoint[i] = 0

    for i in range(num_points):
        ci = arrayPcluster[i]
        arrayCsum[ci, 0] += arrayP[i, 0]
        arrayCsum[ci, 1] += arrayP[i, 1]
        arrayCnumpoint[ci] += 1

    return arrayCsum, arrayCnumpoint


# update the centriods array after computation
@numba.jit(nopython=True, parallel=True, fastmath=True)
def updateCentroids(arrayC, arrayCsum, arrayCnumpoint, num_centroids):
    for i in numba.prange(num_centroids):
        arrayC[i, 0] = arrayCsum[i, 0] / arrayCnumpoint[i]
        arrayC[i, 1] = arrayCsum[i, 1] / arrayCnumpoint[i]


def kmeans(
    arrayP, arrayPcluster, arrayC, arrayCsum, arrayCnumpoint, num_points, num_centroids
):

    for i in range(ITERATIONS):
        with dpctl.device_context(base_kmeans_gpu.get_device_selector()):
            groupByCluster(arrayP, arrayPcluster, arrayC, num_points, num_centroids)

        calCentroidsSum(
            arrayP, arrayPcluster, arrayCsum, arrayCnumpoint, num_points, num_centroids
        )

        updateCentroids(arrayC, arrayCsum, arrayCnumpoint, num_centroids)

    return arrayC, arrayCsum, arrayCnumpoint


def printCentroid(arrayC, arrayCsum, arrayCnumpoint):
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
    #         printCentroid(arrayC, arrayCsum, arrayCnumpoint)

    # print("Iterations: {:d}".format(ITERATIONS))
    # print("Average Time: {:.4f} ms".format(total))


base_kmeans_gpu.run("Kmeans Numba", run_kmeans)
