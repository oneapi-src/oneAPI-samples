#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling SYCL_Essentials Module10 -- SYCL Graphs and Dependencies - 9 of 11 linear_buffers_graphs.cpp
icpx -fsycl lab/linear_buffers_graphs.cpp 
if [ $? -eq 0 ]; then ./a.out; fi

