#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling SYCL_Essentials Module10 -- SYCL Graphs and Dependencies sample - 11 of 11 task_scheduling.cpp
icpx -fsycl lab/task_scheduling.cpp
if [ $? -eq 0 ]; then ./a.out; fi
