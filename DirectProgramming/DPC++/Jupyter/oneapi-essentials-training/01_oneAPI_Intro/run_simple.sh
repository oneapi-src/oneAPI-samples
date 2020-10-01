#!/bin/bash
source /opt/intel/inteloneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling DPCPP_Essentials Module1 -- oneAPI Intro sample - 1 of 2 simple.cpp
dpcpp lab/simple.cpp -o bin/simple
bin/simple
