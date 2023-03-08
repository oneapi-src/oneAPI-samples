#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling SYCL_Essentials Module3 -- SYCL Unified Shared Memory - 2 of 5 usm_explicit.cpp
icpx -fsycl lab/usm_explicit.cpp 
if [ $? -eq 0 ]; then ./a.out; fi

