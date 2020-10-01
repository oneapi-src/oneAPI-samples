#!/bin/bash
source /opt/intel/inteloneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling DPCPP_Essentials Module7 -- oneDPL Intro sample - 3 of 5 dpl_buffer.cpp
export SYCL_BE=PI_OPENCL
dpcpp lab/dpl_buffer.cpp -o dpl_buffer -w
./dpl_buffer
