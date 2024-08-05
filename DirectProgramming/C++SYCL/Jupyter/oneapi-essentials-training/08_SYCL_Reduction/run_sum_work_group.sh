#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling SYCL_Essentials Module8 -- SYCL Reduction - 2 of 8 sum_work_group.cpp
icpx -fsycl lab/sum_work_group.cpp 
if [ $? -eq 0 ]; then ./a.out; fi

