#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling SYCL_Essentials Module12 -- SYCL Atomics Local Memory - 6 of 6 atomics_lab.cpp
icpx -fsycl lab/atomics_lab.cpp 
if [ $? -eq 0 ]; then ./a.out; fi

