#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling SYCL_Essentials Module9 -- SYCL Advanced Buffers and Accessors sample - 6 of 11 accessors_sample.cpp
icpx -fsycl lab/accessors_sample.cpp
if [ $? -eq 0 ]; then ./a.out; fi

