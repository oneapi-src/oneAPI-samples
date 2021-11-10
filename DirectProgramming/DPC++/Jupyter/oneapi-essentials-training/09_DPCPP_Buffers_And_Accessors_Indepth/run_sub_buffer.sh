#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling DPCPP_Essentials Module9 -- DPCPP Advanced Buffers and Accessors sample - 2 of 11 sub_buffers.cpp
dpcpp lab/sub_buffers.cpp -o bin/sub_buffers
if [ $? -eq 0 ]; then bin/sub_buffers; fi

