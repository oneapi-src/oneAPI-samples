#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling SYCL_Essentials Module12 -- SYCL Common Parallel Patterns - 5 of 12 iso2dfd.cpp
icpx -fsycl lab/iso2dfd.cpp
if [ $? -eq 0 ]; then ./a.out 1000 1000 2000; fi

