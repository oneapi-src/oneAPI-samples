# Copyright 2021 Intel Corporation
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

import numpy as np
from numba import vectorize
import dpctl

# UNiversal function that defines element wise for the array and is a numpy operation
@vectorize(nopython=True)
def ufunc_kernel(x, y):
    return x + y


def get_context():
    if dpctl.has_gpu_queues():
        return "opencl:gpu"
    elif dpctl.has_cpu_queues():
        return "opencl:cpu"
    else:
        raise RuntimeError("No device found")


def test_ufunc():
    N = 10
    dtype = np.float64

    A = np.arange(N, dtype=dtype)
    B = np.arange(N, dtype=dtype) * 10

    context = get_context()
    with dpctl.device_context(context):
        C = ufunc_kernel(A, B)

    print(C)


if __name__ == "__main__":
    test_ufunc()
