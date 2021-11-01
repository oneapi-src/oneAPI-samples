#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling DPCPP_Essentials Module9 -- DPCPP Advanced Buffers and Accessors sample - 4 of 11 buffer_set_final_data.cpp
dpcpp lab/buffer_set_final_data.cpp -o bin/buffer_set_final_data
if [ $? -eq 0 ]; then bin/buffer_set_final_data; fi

