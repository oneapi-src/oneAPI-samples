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



from numba import njit, gdb
import array
import random
import dpctl
import timeit

# use njit to offload to a cpu or a gpu

a = array.array('l', [random.randint(0,10) for x in range(0,10000000)])

# use njit to offload to a cpu or a gpu
@njit(parallel=True,fastmath=True)
def f1(x):
    total = 0
    for items in x:
        total +=items
    return total

    
# create a OpenCL queue corresponding to the GPU and call the njit function f1
 
with dpctl.device_context("opencl:gpu:0"):       
    t = timeit.Timer(lambda: f1(a))
    print("Time to calculate the sum in parallel on GPU",t.timeit(200),"seconds")
