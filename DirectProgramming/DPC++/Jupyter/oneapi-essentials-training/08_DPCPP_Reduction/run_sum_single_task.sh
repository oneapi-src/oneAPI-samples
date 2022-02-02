#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling DPCPP_Essentials Module8 -- DPCPP Reduction - 1 of 7 sum_single_task.cpp
dpcpp lab/sum_single_task.cpp 
if [ $? -eq 0 ]; then ./a.out; fi

