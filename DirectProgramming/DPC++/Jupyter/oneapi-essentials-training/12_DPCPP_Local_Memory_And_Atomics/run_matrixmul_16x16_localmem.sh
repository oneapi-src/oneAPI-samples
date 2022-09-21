#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling DPCPP_Essentials Module12 -- DPCPP Atomics Local Memory - 5 of 5 matrixmul_16x16_localmem.cpp
dpcpp lab/matrixmul_16x16_localmem.cpp 
if [ $? -eq 0 ]; then ./a.out; fi

