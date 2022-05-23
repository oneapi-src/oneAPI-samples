#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling DPCPP_Essentials Module10 -- DPCPP Graphs and dependenices - 8 of 11 y_pattern_events.cpp
dpcpp lab/y_pattern_events.cpp
if [ $? -eq 0 ]; then ./a.out; fi

