#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling DPCPP_Essentials Module10 -- DPCPP Graphs and Dependencies - 10 of 10 y_pattern_buffers.cpp
dpcpp lab/y_pattern_buffers.cpp -o bin/y_pattern_buffers
if [ $? -eq 0 ]; then bin/y_pattern_buffers; fi

