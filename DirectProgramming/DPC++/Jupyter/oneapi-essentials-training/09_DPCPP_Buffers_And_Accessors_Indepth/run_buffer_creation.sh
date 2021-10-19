#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling DPCPP_Essentials Module9 -- DPCPP Advanced Buffers and Accessors sample - 1 of 11 buffer_creation.cpp
dpcpp lab/buffer_creation.cpp -o bin/buffer_creation
if [ $? -eq 0 ]; then bin/buffer_creation; fi

