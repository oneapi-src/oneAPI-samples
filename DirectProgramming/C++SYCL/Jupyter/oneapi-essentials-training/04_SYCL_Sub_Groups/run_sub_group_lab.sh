#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling SYCL_Essentials Module4 -- SYCL Sub Groups - 7 of 7 sub_group_lab.cpp
icpx -fsycl lab/sub_group_lab.cpp 
if [ $? -eq 0 ]; then ./a.out; fi


