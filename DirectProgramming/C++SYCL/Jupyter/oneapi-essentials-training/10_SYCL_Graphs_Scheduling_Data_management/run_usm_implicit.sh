#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling SYCL_Essentials Module10 -- SYCL Graphs and Dependencies sample - 2 of 11 USM_implicit.cpp
icpx -fsycl lab/USM_implicit.cpp
if [ $? -eq 0 ]; then ./a.out; fi

