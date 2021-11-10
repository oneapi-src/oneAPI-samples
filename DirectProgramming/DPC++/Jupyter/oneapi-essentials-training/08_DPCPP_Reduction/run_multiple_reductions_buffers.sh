#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling DPCPP_Essentials Module8 -- DPCPP Reduction - 6 of 7 multiple_reductions_buffers.cpp
dpcpp lab/multiple_reductions_buffers.cpp 
if [ $? -eq 0 ]; then ./a.out; fi
