import numpy as np
import numpy.random as rnd

# constants used for input data generation
SEED = 7777777

# write input data to a file in binary format
def __dump_binary__(X, Y):
    with open("X.bin", "w") as fd:
        X.tofile(fd)

    with open("Y.bin", "w") as fd:
        Y.tofile(fd)


# write input data to a file in text format
def __dump_text__(X, Y):
    with open("X.txt", "w") as fd:
        X.tofile(fd, "\n", "%s")

    with open("Y.txt", "w") as fd:
        Y.tofile(fd, "\n", "%s")


# call numpy.random.uniform to generate input data
def gen_rand_data(nopt, dims, dtype=np.float64):
    rnd.seed(SEED)
    return (
        rnd.random((nopt, dims)).astype(dtype),
        rnd.random((nopt, dims)).astype(dtype),
    )


# call numpy.random.uniform to generate input data and write the input as binary to a file
def gen_data_to_file(nopt, dims, dtype=np.float64):
    X, Y = gen_rand_data(nopt, dims, dtype)
    __dump_binary__(X, Y)
    # __dump_text__(X, Y) #for verification purpose only