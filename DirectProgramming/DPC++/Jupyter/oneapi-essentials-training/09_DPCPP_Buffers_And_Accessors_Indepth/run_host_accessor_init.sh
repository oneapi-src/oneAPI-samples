#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling DPCPP_Essentials Module9 -- DPCPP Advanced Buffers and Accessors sample - 7 of 10 host_accessor_init.cpp
dpcpp lab/host_accessor_init.cpp
if [ $? -eq 0 ]; then ./a.out; fi

