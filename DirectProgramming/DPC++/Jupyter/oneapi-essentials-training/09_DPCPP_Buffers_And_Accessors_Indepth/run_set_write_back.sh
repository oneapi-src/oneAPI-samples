#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling DPCPP_Essentials Module9 -- DPCPP Advanced Buffers and Accessors sample - 5 of 11 buffer_set_write_back.cpp
dpcpp lab/buffer_set_write_back.cpp -o bin/buffer_set_write_back
if [ $? -eq 0 ]; then bin/buffer_set_write_back; fi

