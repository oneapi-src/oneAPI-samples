#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling DPCPP_Essentials Module12 -- DPCPP Atomics Local Memory - 6 of 6 atomics_lab.cpp
dpcpp lab/atomics_lab.cpp 
if [ $? -eq 0 ]; then ./a.out; fi

