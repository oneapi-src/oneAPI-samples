#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling SYCL_Essentials Module9 -- SYCL Advanced Buffers and Accessors sample - 6 of 10 host_accessor_sample.cpp
icpx -fsycl lab/host_accessor_sample.cpp
if [ $? -eq 0 ]; then ./a.out; fi

