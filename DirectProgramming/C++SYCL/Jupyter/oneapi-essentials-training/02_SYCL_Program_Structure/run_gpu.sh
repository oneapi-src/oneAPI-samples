#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling SYCL_Essentials Module2 -- SYCL Program Structure sample - 1 of 7 gpu_sample.cpp
icpx -fsycl lab/gpu_sample.cpp
if [ $? -eq 0 ]; then ./a.out; fi

