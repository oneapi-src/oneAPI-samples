#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling SYCL_Essentials Module12 -- SYCL Common Parallel Patterns - 4 of 12 stencil_localmem.cpp
icpx -fsycl lab/stencil_localmem.cpp
if [ $? -eq 0 ]; then ./a.out; fi

