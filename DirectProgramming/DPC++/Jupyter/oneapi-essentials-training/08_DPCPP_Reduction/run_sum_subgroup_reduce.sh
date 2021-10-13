#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling DPCPP_Essentials Module8 -- DPCPP Reduction - 3 of 6 sum_subgroup_reduce.cpp
dpcpp lab/sum_subgroup_reduce.cpp 
if [ $? -eq 0 ]; then ./a.out; fi

