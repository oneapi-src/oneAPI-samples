#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling SYCL_Essentials Module12 -- SYCL Common Parallel Patterns - 3 of 12 stencil.cpp
icpx -fsycl lab/stencil.cpp
if [ $? -eq 0 ]; then ./a.out; fi

