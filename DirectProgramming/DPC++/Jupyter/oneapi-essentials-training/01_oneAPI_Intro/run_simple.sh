#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling DPCPP_Essentials Module1 -- oneAPI Intro sample - 1 of 2 simple.cpp
dpcpp lab/simple.cpp -o bin/simple
if [ $? -eq 0 ]; then bin/simple; fi

