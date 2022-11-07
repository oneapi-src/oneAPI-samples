#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling DPCPP_Essentials Module7 -- oneDPL Intro sample - 1 of 5 dpl_simple.cpp
dpcpp lab/dpl_simple.cpp -o dpl_simple
if [ $? -eq 0 ]; then ./dpl_simple; fi

