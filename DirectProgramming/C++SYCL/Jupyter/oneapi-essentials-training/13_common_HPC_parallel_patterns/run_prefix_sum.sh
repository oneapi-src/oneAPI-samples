#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling SYCL_Essentials Module12 -- SYCL Common Parallel Patterns - 6 of 12 prefix_sum.cpp
icpx -fsycl lab/prefix_sum.cpp 
if [ $? -eq 0 ]; then ./a.out 21 47; fi

