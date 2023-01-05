#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling SYCL_Essentials Module3 -- SYCL Unified Shared Memory - 1 of 5 usm.cpp
icpx -fsycl lab/usm.cpp 
if [ $? -eq 0 ]; then ./a.out; fi

