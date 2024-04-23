#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
icpx -fsycl lab/sycl_vector_add.cpp 
if [ $? -eq 0 ]; then ./a.out; fi
