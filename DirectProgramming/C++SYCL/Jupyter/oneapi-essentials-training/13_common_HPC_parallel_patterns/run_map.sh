#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling SYCL_Essentials Module12 -- SYCL Common Parallel Patterns - 2 of 12 basic_map.cpp
icpx -fsycl lab/basic_map.cpp 
if [ $? -eq 0 ]; then ./a.out; fi

