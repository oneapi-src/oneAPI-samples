#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling SYCL_Essentials Module7 -- oneDPL Intro sample - 3 of 5 dpl_buffer.cpp
#export SYCL_BE=PI_OPENCL
icpx -fsycl lab/dpl_buffer.cpp -o dpl_buffer -w
if [ $? -eq 0 ]; then ./dpl_buffer; fi

