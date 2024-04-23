#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling SYCL_Essentials Module8 -- SYCL Reduction - 3 of 8 sum_workgroup_reduce.cpp
icpx -fsycl lab/sum_workgroup_reduce.cpp
if [ $? -eq 0 ]; then ./a.out; fi

