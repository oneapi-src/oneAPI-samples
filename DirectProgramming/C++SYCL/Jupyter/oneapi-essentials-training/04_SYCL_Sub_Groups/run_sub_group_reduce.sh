#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling SYCL_Essentials Module4 -- SYCL Sub Groups - 4 of 7 sub_group_reduce.cpp
icpx -fsycl lab/sub_group_reduce.cpp 
if [ $? -eq 0 ]; then ./a.out; fi


