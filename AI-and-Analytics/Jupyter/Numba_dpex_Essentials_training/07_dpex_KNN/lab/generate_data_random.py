import dpnp as np

try:
    import dpnp.random_intel as rnd

    numpy_ver = "Intel"
except:
    import numpy.random as rnd

    numpy_ver = "regular"

# constants used for input data generation
SEED_TEST = 777777
SEED_TRAIN = 0
DATA_DIM = 16
CLASSES_NUM = 3
TRAIN_DATA_SIZE = 2**10
N_NEIGHBORS = 5

# write input data to a file in binary format
def __dump_binary__(data_array, file_name):
    with open(file_name, "w") as fd:
        data_array.tofile(fd)


# write input data to a file in text format
def __dump_text__(data_array, file_name):
    with open(file_name, "w") as fd:
        data_array.tofile(fd, "\n", "%s")


def __gen_data_x__(ip_size, dtype, seed):
    rnd.seed(seed)
    data = rnd.rand(ip_size, DATA_DIM)
    return data.astype(dtype)


def __gen_data_y__(ip_size, seed):
    rnd.seed(seed)
    data = rnd.randint(CLASSES_NUM, size=ip_size)
    return data


def gen_train_data(dtype=np.float64):
    return (
        __gen_data_x__(TRAIN_DATA_SIZE, dtype, SEED_TRAIN),
        __gen_data_y__(TRAIN_DATA_SIZE, SEED_TRAIN),
    )


def gen_test_data(ip_size, dtype=np.float64):
    return __gen_data_x__(ip_size, dtype, SEED_TEST)


# call numpy.random.uniform to generate input data and write the input as binary to a file
def gen_data_to_file(ip_size, dtype=np.float64):
    x_train, y_train = gen_train_data(dtype)
    x_test = gen_test_data(ip_size, dtype)

    __dump_binary__(x_train, "x_train.bin")
    __dump_binary__(y_train, "y_train.bin")
    __dump_binary__(x_test, "x_test.bin")