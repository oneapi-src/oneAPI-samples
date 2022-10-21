import numpy as np
import numpy.random as rnd

# constants used for input data generation
SEED = 777777
DTYPE = np.float32


# write input data to a file in binary format
def __dump_binary__(data_array, file_name):
    with open(file_name, "w") as fd:
        data_array.tofile(fd)


# write input data to a file in text format
def __dump_text__(data_array, file_name):
    with open(file_name, "w") as fd:
        data_array.tofile(fd, "\n", "%s")


def gen_data(nopt, dims, dtype=DTYPE):
    rnd.seed(SEED)
    return (
        rnd.random((nopt, dims)).astype(dtype),
        rnd.random((nopt, dims)).astype(dtype),
    )


# call numpy.random.uniform to generate input data and write the input as binary to a file
def gen_data_to_file(nopt, dims, dtype=DTYPE):
    x_data, y_data = gen_data(nopt, dims, dtype)
    __dump_binary__(x_data, "x_data.bin")
    __dump_binary__(y_data, "y_data.bin")
    #__dump_text__(x_data, "x_data.bin")
    #__dump_text__(y_data, "y_data.bin")