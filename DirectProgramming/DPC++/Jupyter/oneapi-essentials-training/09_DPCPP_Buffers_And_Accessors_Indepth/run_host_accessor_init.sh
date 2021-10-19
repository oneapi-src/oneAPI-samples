#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling DPCPP_Essentials Module9 -- DPCPP Advanced Buffers and Accessors sample - 3 of 11 host_accessor_init.cpp
dpcpp lab/host_accessor_init.cpp -o bin/host_accessor_init
if [ $? -eq 0 ]; then bin/host_accessor_init; fi

