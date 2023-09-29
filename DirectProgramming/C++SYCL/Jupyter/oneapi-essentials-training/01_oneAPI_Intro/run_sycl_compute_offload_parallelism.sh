#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
icpx -fsycl lab/sycl_compute_offload_parallelism.cpp 
if [ $? -eq 0 ]; then ./a.out; fi
