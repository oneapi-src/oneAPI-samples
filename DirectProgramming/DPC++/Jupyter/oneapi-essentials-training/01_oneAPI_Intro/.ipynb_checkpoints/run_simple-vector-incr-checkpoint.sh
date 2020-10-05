#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling DPCPP_Essentials Module1 -- oneAPI Intro sample - 2 of 2 simple-vector-incr.cpp
dpcpp lab/simple-vector-incr.cpp -o bin/simple-vector-incr
if [ $? -eq 0 ]; then bin/simple-vector-incr; fi
