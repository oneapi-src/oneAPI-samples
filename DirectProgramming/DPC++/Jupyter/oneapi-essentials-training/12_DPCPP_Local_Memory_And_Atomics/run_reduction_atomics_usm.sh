#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling DPCPP_Essentials Module12 -- DPCPP Atomics Local Memory - 1 of 5 reduction_atomics_usm.cpp
dpcpp lab/reduction_atomics_usm.cpp 
if [ $? -eq 0 ]; then ./a.out; fi

