import numpy as np
import numpy.random as rnd

#constants used for input data generation
DEFAULT_NBINS = 20
DEFAULT_RMIN, DEFAULT_RMAX = 0.1, 50

#write input data to a file in binary format
def __dump_binary__(x1, y1, z1, w1, x2, y2, z2, w2, DEFAULT_RBINS_SQUARED):
    with open('x1.bin', 'w') as fd:
        x1.tofile(fd)

    with open('y1.bin', 'w') as fd:
        y1.tofile(fd)

    with open('z1.bin', 'w') as fd:
        z1.tofile(fd)

    with open('w1.bin', 'w') as fd:
        w1.tofile(fd)

    with open('x2.bin', 'w') as fd:
        x2.tofile(fd)

    with open('y2.bin', 'w') as fd:
        y2.tofile(fd)

    with open('z2.bin', 'w') as fd:
        z2.tofile(fd)

    with open('w2.bin', 'w') as fd:
        w2.tofile(fd)

    with open('DEFAULT_RBINS_SQUARED.bin', 'w') as fd:
        DEFAULT_RBINS_SQUARED.tofile(fd)

#write input data to a file in text format
def __dump_text__(x1, y1, z1, w1, x2, y2, z2, w2, DEFAULT_RBINS_SQUARED):
    with open('x1.txt', 'w') as fd:
        x1.tofile(fd, '\n', '%s')

    with open('y1.txt', 'w') as fd:
        y1.tofile(fd, '\n', '%s')

    with open('z1.txt', 'w') as fd:
        z1.tofile(fd, '\n', '%s')

    with open('w1.txt', 'w') as fd:
        w1.tofile(fd, '\n', '%s')

    with open('x2.txt', 'w') as fd:
        x2.tofile(fd, '\n', '%s')

    with open('y2.txt', 'w') as fd:
        y2.tofile(fd, '\n', '%s')

    with open('z2.txt', 'w') as fd:
        z2.tofile(fd, '\n', '%s')

    with open('w2.txt', 'w') as fd:
        w2.tofile(fd, '\n', '%s')

    with open('DEFAULT_RBINS_SQUARED.txt', 'w') as fd:
        DEFAULT_RBINS_SQUARED.tofile(fd, '\n', '%s')

def __random_weighted_points__(n, Lbox, seed, dtype):
    rng = rnd.RandomState(seed)
    data = rng.uniform(0, 1, n*4)
    x, y, z, w = (
        data[:n]*Lbox, data[n:2*n]*Lbox, data[2*n:3*n]*Lbox, data[3*n:])
    return (
        x.astype(dtype), y.astype(dtype), z.astype(dtype),
        w.astype(dtype))

def __generate_rbins__(dtype):
    DEFAULT_RBINS = np.logspace(np.log10(DEFAULT_RMIN), np.log10(DEFAULT_RMAX), DEFAULT_NBINS).astype(dtype)
    DEFAULT_RBINS_SQUARED = (DEFAULT_RBINS**2).astype(dtype)

    return DEFAULT_RBINS_SQUARED

# call numpy to generate input data
def gen_rand_data(npoints, dtype = np.float64):
    Lbox = 500.
    n1 = npoints
    n2 = npoints
    x1, y1, z1, w1 = __random_weighted_points__(n1, Lbox, 0, dtype)
    x2, y2, z2, w2 = __random_weighted_points__(n2, Lbox, 1, dtype)

    DEFAULT_RBINS_SQUARED = __generate_rbins__(dtype)

    return (
        x1, y1, z1, w1, x2, y2, z2, w2, DEFAULT_RBINS_SQUARED
    )

# call numpy.random.uniform to generate input data and write the input as binary to a file
def gen_data_to_file(npoints, dtype = np.float64):
    x1, y1, z1, w1, x2, y2, z2, w2, DEFAULT_RBINS_SQUARED = gen_rand_data(npoints, dtype)
    __dump_binary__(x1, y1, z1, w1, x2, y2, z2, w2, DEFAULT_RBINS_SQUARED)
    #__dump_text__(x1, y1, z1, w1, x2, y2, z2, w2, DEFAULT_RBINS_SQUARED) #for verification purpose only