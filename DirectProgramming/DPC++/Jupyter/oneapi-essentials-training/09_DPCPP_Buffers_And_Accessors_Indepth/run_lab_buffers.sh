#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling DPCPP_Essentials Module9 -- DPCPP Advanced Buffers and Accessors sample - 10 of 10 lab_buffers.cpp
dpcpp lab/lab_buffers.cpp
if [ $? -eq 0 ]; then ./a.out; fi

