#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling DPCPP_Essentials Module9 -- DPCPP Advanced Buffers and Accessors sample - 3 of 10 buffer_set_final_data.cpp
dpcpp lab/buffer_set_final_data.cpp
if [ $? -eq 0 ]; then ./a.out; fi

