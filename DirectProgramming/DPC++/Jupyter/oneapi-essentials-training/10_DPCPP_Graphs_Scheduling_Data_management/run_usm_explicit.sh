#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling DPCPP_Essentials Module10 -- DPCPP Graphs and Dependencies - 1 of 10 USM_explicit.cpp
dpcpp lab/USM_explicit.cpp -o bin/USM_explicit
if [ $? -eq 0 ]; then bin/USM_explicit; fi

