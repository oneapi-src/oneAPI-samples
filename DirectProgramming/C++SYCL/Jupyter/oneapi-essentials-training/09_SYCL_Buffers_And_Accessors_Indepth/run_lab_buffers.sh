#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling SYCL_Essentials Module9 -- SYCL Advanced Buffers and Accessors sample - 10 of 10 lab_buffers.cpp
icpx -fsycl lab/lab_buffers.cpp
if [ $? -eq 0 ]; then ./a.out; fi

