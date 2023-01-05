#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling SYCL_Essentials Module3 -- SYCL Unified Shared Memory - 4 of 5 usm_data2.cpp
icpx -fsycl lab/usm_data2.cpp 
if [ $? -eq 0 ]; then ./a.out; fi

