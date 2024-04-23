#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling SYCL_Essentials Module12 -- SYCL Atomics Local Memory - 1 of 5 reduction_atomics_buffer.cpp
icpx -fsycl lab/reduction_atomics_buffer.cpp 
if [ $? -eq 0 ]; then ./a.out; fi

