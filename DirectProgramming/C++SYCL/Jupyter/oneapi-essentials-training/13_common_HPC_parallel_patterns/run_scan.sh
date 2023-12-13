#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling SYCL_Essentials Module12 -- SYCL Common Parallel Patterns - 7 of 12 basic_scan.cpp
icpx -fsycl lab/basic_scan.cpp 
if [ $? -eq 0 ]; then ./a.out; fi

