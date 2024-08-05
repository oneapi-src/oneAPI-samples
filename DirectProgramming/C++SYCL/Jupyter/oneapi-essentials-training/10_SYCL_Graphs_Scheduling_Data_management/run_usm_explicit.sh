#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling SYCL_Essentials Module10 -- SYCL Graphs and Dependencies - 1 of 11 USM_explicit.cpp
icpx -fsycl lab/USM_explicit.cpp
if [ $? -eq 0 ]; then ./a.out; fi

