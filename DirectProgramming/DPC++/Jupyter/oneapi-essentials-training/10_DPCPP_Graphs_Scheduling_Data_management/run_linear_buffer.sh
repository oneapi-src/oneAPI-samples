#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling DPCPP_Essentials Module10 -- DPCPP Graphs and Dependencies - 9 of 10 linear_buffers_graphs.cpp
dpcpp lab/linear_buffers_graphs.cpp -o bin/linear_buffers_graphs
if [ $? -eq 0 ]; then bin/linear_buffers_graphs; fi

