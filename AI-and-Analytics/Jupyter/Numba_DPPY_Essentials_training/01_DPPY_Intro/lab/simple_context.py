# Copyright 2020, 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
The numba_dppy extension adds an automatic offload optimizer to
numba. The optimizer automatically detects data-parallel code
regions in a numba.jit function and then offloads the data-parallel
regions to a SYCL device. The optimizer is triggered when a numba.jit
function is invoked inside a dpctl ``device_context`` scope.
This example demonstrates the usage of numba_dppy's automatic offload
functionality. Note that numba_dppy should be installed in your
environment for the example to work.
"""

import numpy as np
from numba import njit, prange
import dpctl


@njit
def add_two_arrays(b, c):
    a = np.empty_like(b)
    for i in prange(len(b)):
        a[i] = b[i] + c[i]

    return a

def main():
    N = 10
    b = np.ones(N)
    c = np.ones(N)

    # Use the environment variable SYCL_DEVICE_FILTER to change the default device.
    # See https://github.com/intel/llvm/blob/sycl/sycl/doc/EnvironmentVariables.md#sycl_device_filter.
    device = dpctl.select_default_device()
    print("Using device ...")
    device.print_device_info()

    with dpctl.device_context(device):
        result = add_two_arrays(b, c)

    print("Result :", result)

    print("Done...")


if __name__ == "__main__":
    main()
    
