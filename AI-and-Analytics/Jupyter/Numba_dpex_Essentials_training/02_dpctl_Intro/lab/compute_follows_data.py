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

import warnings

import dpctl
import dpctl.tensor as dpt
import numpy as np

import numba_dpex

@numba_dpex.kernel
def sum_kernel(a, b, c):
    i = numba_dpex.get_global_id(0)
    c[i] = a[i] + b[i]


def allocate_SUAI_data(a, b, got, usm_type, queue):
    da = dpt.usm_ndarray(
        a.shape,
        dtype=a.dtype,
        buffer=usm_type,
        buffer_ctor_kwargs={"queue": queue},
    )
    da.usm_data.copy_from_host(a.reshape((-1)).view("|u1"))

    db = dpt.usm_ndarray(
        b.shape,
        dtype=b.dtype,
        buffer=usm_type,
        buffer_ctor_kwargs={"queue": queue},
    )
    db.usm_data.copy_from_host(b.reshape((-1)).view("|u1"))

    dc = dpt.usm_ndarray(
        got.shape,
        dtype=got.dtype,
        buffer=usm_type,
        buffer_ctor_kwargs={"queue": queue},
    )

    return da, db, dc


# ==========================================================================

def select_device_SUAI(N):
    usm_type = "device"

    a = np.array(np.random.random(N), np.float32)
    b = np.array(np.random.random(N), np.float32)
    got = np.ones_like(a)

    device = dpctl.SyclDevice("gpu")
    queue = dpctl.SyclQueue(device)

    # We are allocating the data in Opencl GPU and this device
    # will be selected for compute.
    
    da, db, dc = allocate_SUAI_data(a, b, got, usm_type, queue)    

    # Users don't need to specify where the computation will
    # take place. It will be inferred from data.
    sum_kernel[N, 1](da, db, dc)    

    dc.usm_data.copy_to_host(got.reshape((-1)).view("|u1"))   

    expected = a + b    
    print(
        "Correct result when array with __sycl_usm_array_interface__ is passed!"
    )


if __name__ == "__main__":    
    select_device_SUAI(10)
