#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling DPCPP_Essentials Module10 -- DPCPP Graphs and dependenices - 7 of 11 linear_event_graphs.cpp
dpcpp lab/linear_event_graphs.cpp
if [ $? -eq 0 ]; then ./a.out; fi

