#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
cd sycl_migrated/Samples/4_CUDA_Libraries/matrixMulCUBLAS
icpx -fsycl -I ../../../Common -I ../../../include *.cpp -lmkl_sycl -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -w
if [ $? -eq 0 ]; then ./a.out; fi

