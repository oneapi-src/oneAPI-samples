#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling SYCL_Essentials Module7 -- oneDPL Intro sample - 5 of 5 dpl_usm_alloc.cpp
icpx -fsycl lab/dpl_usm_alloc.cpp -o dpl_usm_alloc -w
if [ $? -eq 0 ]; then ./dpl_usm_alloc; fi

