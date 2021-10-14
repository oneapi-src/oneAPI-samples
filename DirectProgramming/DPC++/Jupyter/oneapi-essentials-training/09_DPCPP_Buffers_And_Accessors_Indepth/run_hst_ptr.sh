#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling DPCPP_Essentials Module9 -- DPCPP Advanced Buffers and Accessors sample - 3 of 11 buffer_host_ptr.cpp
dpcpp lab/buffer_host_ptr.cpp -o bin/buffer_host_ptr
if [ $? -eq 0 ]; then bin/buffer_host_ptr; fi

