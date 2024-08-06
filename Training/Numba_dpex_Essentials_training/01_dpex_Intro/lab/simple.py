
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

from numba import njit
import numpy as np
import dpctl
import timeit

@njit(parallel=False)
def f1(a, b,c,N):
   for i in range(N):    
    c[i] = a[i] + b[i]    
    

N = 1000000000
a = np.ones(N, dtype=np.float32)
b = np.ones(N, dtype=np.float32)
c = np.zeros(N,dtype=np.float32)

t = timeit.Timer(lambda: f1(a,b,c,N))
print("Time to calculate the sum in parallel on GPU",t.timeit(200),"seconds")
print(c)
    
