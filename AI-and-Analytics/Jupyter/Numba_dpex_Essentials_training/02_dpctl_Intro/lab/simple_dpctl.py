
#                      Data Parallel Control (dpctl)
#
# Copyright 2020-2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Demonstrates host to device copy functions using dpctl.memory.
"""

import numpy as np

import dpctl.memory as dpmem

ms = dpmem.MemoryUSMShared(32)
md = dpmem.MemoryUSMDevice(32)

host_buf = np.random.randint(0, 42, dtype=np.uint8, size=32)

# copy host byte-like object to USM-device buffer
md.copy_from_host(host_buf)

# copy USM-device buffer to USM-shared buffer in parallel using
# sycl::queue::memcpy.
ms.copy_from_device(md)

# build numpy array reusing host-accessible USM-shared memory
X = np.ndarray((len(ms),), buffer=ms, dtype=np.uint8)

# Display Python object NumPy ndarray is viewing into
print("numpy.ndarray.base: ", X.base)
print("")

# Print content of the view
print("View..........: ", X)

# Print content of the original host buffer
print("host_buf......: ", host_buf)

# use copy_to_host to retrieve memory of USM-device memory
print("copy_to_host(): ", md.copy_to_host())

