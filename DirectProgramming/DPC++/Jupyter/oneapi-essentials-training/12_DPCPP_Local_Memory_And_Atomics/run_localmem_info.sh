#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling DPCPP_Essentials Module12 -- DPCPP Atomics Local Memory - 3 of 5 localmem_info.cpp
dpcpp lab/localmem_info.cpp 
if [ $? -eq 0 ]; then ./a.out; fi

