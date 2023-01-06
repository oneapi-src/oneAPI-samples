#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling SYCL_Essentials Module3 -- SYCL Unified Shared Memory - 5 of 5 usm_lab.cpp
icpx -fsycl lab/usm_lab.cpp 
if [ $? -eq 0 ]; then ./a.out; fi

