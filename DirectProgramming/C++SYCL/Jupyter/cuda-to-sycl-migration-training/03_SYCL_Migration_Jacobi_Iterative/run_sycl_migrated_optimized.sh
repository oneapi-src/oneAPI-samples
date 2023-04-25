#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
cd sycl_migrated_optimized/Samples/3_CUDA_Features/jacobiCudaGraphs/
icpx -fsycl -I ../../../Common -I ../../../include *.cpp -w
if [ $? -eq 0 ]; then ./a.out; fi
