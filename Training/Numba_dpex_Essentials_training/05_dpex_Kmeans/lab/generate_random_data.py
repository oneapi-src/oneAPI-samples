import numpy as np

try:
    import numpy.random_intel as rnd
except:
    import numpy.random as rnd

# constants used for input data generation
SEED = 7777777
XL = 1.0
XH = 5.0

# write input data to a file in binary format
def __dump_binary__(X, arrayPclusters, arrayC, arrayCsum, arrayCnumpoint):
    with open("X.bin", "w") as fd:
        X.tofile(fd)

    with open("arrayPclusters.bin", "w") as fd:
        arrayPclusters.tofile(fd)

    # with open('arrayC.bin', 'w') as fd:
    #     arrayC.tofile(fd)

    # with open('arrayCsum.bin', 'w') as fd:
    #     arrayCsum.tofile(fd)

    # with open('arrayCnumpoint.bin', 'w') as fd:
    #     arrayCnumpoint.tofile(fd)


# write input data to a file in text format
def __dump_text__(X, arrayPclusters, arrayC, arrayCsum, arrayCnumpoint):
    with open("X.txt", "w") as fd:
        X.tofile(fd, "\n", "%s")

    with open("arrayPclusters.txt", "w") as fd:
        arrayPclusters.tofile(fd, "\n", "%s")

    # with open('arrayC.txt', 'w') as fd:
    #     arrayC.tofile(fd, '\n', '%s')

    # with open('arrayCsum.txt', 'w') as fd:
    #     arrayCsum.tofile(fd, '\n', '%s')

    # with open('arrayCnumpoint.txt', 'w') as fd:
    #     arrayCnumpoint.tofile(fd, '\n', '%s')


# call numpy.random.uniform to generate input data
def gen_rand_data(nopt, dims=2, NUMBER_OF_CENTROIDS=10, dtype=np.float64):
    rnd.seed(SEED)
    return (
        rnd.uniform(XL, XH, (nopt, dims)).astype(dtype),
        np.ones(nopt, dtype=np.int32),
        np.ones((NUMBER_OF_CENTROIDS, 2), dtype=dtype),
        np.ones((NUMBER_OF_CENTROIDS, 2), dtype=dtype),
        np.ones(NUMBER_OF_CENTROIDS, dtype=np.int32),
    )


# call numpy.random.uniform to generate input data and write the input as binary to a file
def gen_data_to_file(nopt, dims=2, NUMBER_OF_CENTROIDS=10, dtype=np.float64):
    X, arrayPclusters, arrayC, arrayCsum, arrayCnumpoint = gen_rand_data(
        nopt, dims, NUMBER_OF_CENTROIDS, dtype
    )
    __dump_binary__(X, arrayPclusters, arrayC, arrayCsum, arrayCnumpoint)
    # __dump_text__(X,arrayPclusters,arrayC,arrayCsum,arrayCnumpoint) #for verification purpose only