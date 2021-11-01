#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling DPCPP_Essentials Module10 -- DPCPP Graphs and dependenices - 7 of 10 linear_event_graphs.cpp
dpcpp lab/linear_event_graphs.cpp -o bin/linear_event_graphs
if [ $? -eq 0 ]; then bin/linear_event_graphs; fi

