
from numba import njit,prange
import numpy as np
import dpctl
import timeit

@njit(parallel=True)
def f1(a, b,c,N):
   for i in prange(N):    
    c[i] = a[i] + b[i]     


N = 500000
a = np.ones(N, dtype=np.float32)
b = np.ones(N, dtype=np.float32)
c = np.zeros(N,dtype=np.float32)

t = timeit.Timer(lambda: f1(a,b,c,N))
print("Time to calculate the sum in parallel",t.timeit(200),"seconds")
print(c)
