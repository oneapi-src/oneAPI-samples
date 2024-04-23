#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling SYCL_Essentials Module10 -- SYCL Graphs and Dependencies sample - 4 of 11 accessors_WAR_WAW.cpp
icpx -fsycl lab/accessors_WAR_WAW.cpp
if [ $? -eq 0 ]; then ./a.out; fi

