#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling SYCL_Essentials Module9 -- SYCL Advanced Buffers and Accessors sample - 2 of 10 buffer_host_ptr.cpp
icpx -fsycl lab/buffer_host_ptr.cpp
if [ $? -eq 0 ]; then ./a.out; fi

