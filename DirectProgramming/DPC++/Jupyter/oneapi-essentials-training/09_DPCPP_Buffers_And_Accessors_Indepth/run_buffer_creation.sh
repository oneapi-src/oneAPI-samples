#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling DPCPP_Essentials Module9 -- DPCPP Advanced Buffers and Accessors sample - 1 of 10 buffer_creation.cpp
dpcpp lab/buffer_creation.cpp
if [ $? -eq 0 ]; then ./a.out; fi

