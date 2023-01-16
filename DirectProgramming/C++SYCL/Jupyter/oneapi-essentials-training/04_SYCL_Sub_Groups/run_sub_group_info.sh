#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling SYCL_Essentials Module4 -- SYCL Sub Groups - 1 of 7 sub_group_info.cpp
icpx -fsycl lab/sub_group_info.cpp 
if [ $? -eq 0 ]; then ./a.out; fi


