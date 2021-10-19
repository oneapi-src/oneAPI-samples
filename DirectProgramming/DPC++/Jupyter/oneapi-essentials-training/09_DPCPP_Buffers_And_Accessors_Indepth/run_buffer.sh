#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling DPCPP_Essentials Module9 -- DPCPP Advanced Buffers and Accessors sample - 7 of 11 buffer_sample.cpp
dpcpp lab/buffer_sample.cpp -o bin/buffer_sample
if [ $? -eq 0 ]; then bin/buffer_sample; fi

