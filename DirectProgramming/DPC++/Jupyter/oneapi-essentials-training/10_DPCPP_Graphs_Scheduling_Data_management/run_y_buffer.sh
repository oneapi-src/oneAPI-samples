#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling DPCPP_Essentials Module10 -- DPCPP Graphs and Dependencies - 10 of 11 y_pattern_buffers.cpp
dpcpp lab/y_pattern_buffers.cpp
if [ $? -eq 0 ]; then ./a.out; fi

