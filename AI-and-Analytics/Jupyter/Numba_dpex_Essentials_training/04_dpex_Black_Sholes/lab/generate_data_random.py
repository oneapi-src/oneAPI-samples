import numpy as np

try:
    import numpy.random_intel as rnd
except:
    import numpy.random as rnd

# constants used for input data generation
SEED = 777777
S0L = 10.0
S0H = 50.0
XL = 10.0
XH = 50.0
TL = 1.0
TH = 2.0

# write input data to a file in binary format
def __dump_binary__(price, strike, t):
    with open("price.bin", "w") as fd:
        price.tofile(fd)

    with open("strike.bin", "w") as fd:
        strike.tofile(fd)

    with open("t.bin", "w") as fd:
        t.tofile(fd)


# write input data to a file in text format
def __dump_text__(price, strike, t):
    with open("gen_price.txt", "w") as fd:
        price.tofile(fd, "\n", "%s")

    with open("gen_strike.txt", "w") as fd:
        strike.tofile(fd, "\n", "%s")

    with open("gen_t.txt", "w") as fd:
        t.tofile(fd, "\n", "%s")


# call numpy.random.uniform to generate input data
def gen_rand_data(nopt, dtype=np.float64):
    rnd.seed(SEED)
    return (
        rnd.uniform(S0L, S0H, nopt).astype(dtype),
        rnd.uniform(XL, XH, nopt).astype(dtype),
        #np.linspace(S0L, S0H, nopt).astype(dtype),
        #np.linspace(XL, XH, nopt).astype(dtype),
        rnd.uniform(TL, TH, nopt).astype(dtype),
    )


# call numpy.random.uniform to generate input data and write the input as binary to a file
def gen_data_to_file(nopt, dtype=np.float64):
    price, strike, t = gen_rand_data(nopt, dtype)
    __dump_binary__(price, strike, t)
    # __dump_text__(price, strike, t) #for verification purpose only