#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling SYCL_Essentials Module12 -- SYCL Common Parallel Patterns - 9 of 12 array_reductions.cpp
icpx -fsycl lab/array_reductions.cpp
if [ $? -eq 0 ]; then ./a.out; fi

