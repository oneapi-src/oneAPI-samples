#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling DPCPP_Essentials Module9 -- DPCPP Advanced Buffers and Accessors sample - 6 of 11 accessors_sample.cpp
dpcpp lab/accessors_sample.cpp
if [ $? -eq 0 ]; then ./a.out; fi

