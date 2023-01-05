#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling SYCL_Essentials Module1 -- oneAPI Intro sample - 1 of 1 simple.cpp
icpx -fsycl lab/simple.cpp
if [ $? -eq 0 ]; then ./a.out; fi

