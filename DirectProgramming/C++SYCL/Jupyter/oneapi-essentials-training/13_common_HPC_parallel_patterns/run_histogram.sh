#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling SYCL_Essentials Module12 -- SYCL Common Parallel Patterns - 10 of 12 histogram.cpp
icpx -fsycl lab/histogram.cpp
if [ $? -eq 0 ]; then ./a.out; fi

