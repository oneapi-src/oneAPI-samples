#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling DPCPP_Essentials Module10 -- DPCPP Graphs and Dependencies sample - 2 of 10 USM_implicit.cpp
dpcpp lab/USM_implicit.cpp -o bin/USM_implicit
if [ $? -eq 0 ]; then bin/USM_implicit; fi

