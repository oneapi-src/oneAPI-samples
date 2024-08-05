#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling SYCL_Essentials Module12 -- SYCL Common Parallel Patterns - 11 of 12 local_pack.cpp
icpx -fsycl lab/local_pack.cpp
if [ $? -eq 0 ]; then ./a.out; fi

