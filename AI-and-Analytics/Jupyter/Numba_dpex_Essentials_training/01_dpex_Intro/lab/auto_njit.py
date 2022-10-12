
import dpctl
import numpy as np
import numba
from numba import njit, prange

@numba.njit(parallel=True)
def l2_distance_kernel(a, b):
    sub = a - b
    sq = np.square(sub)
    sum = np.sum(sq)
    d = np.sqrt(sum)
    return d

def main():
    R = 64
    C = 1
    
    X = np.random.random((R,C))
    Y = np.random.random((R,C))
    
    device = dpctl.select_default_device()
    print("Using device ...")
    device.print_device_info()

    with dpctl.device_context(device):
        result = l2_distance_kernel(X, Y)

    print("Result :", result)
    print("Done...")

if __name__ == "__main__":
    main()
