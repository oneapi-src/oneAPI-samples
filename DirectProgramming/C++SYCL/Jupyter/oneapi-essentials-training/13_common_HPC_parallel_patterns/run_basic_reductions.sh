#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling SYCL_Essentials Module12 -- SYCL Common Parallel Patterns - 8 of 12 basic_reductions.cpp
icpx -fsycl lab/basic_reductions.cpp
if [ $? -eq 0 ]; then ./a.out; fi

