#!/bin/bash
source /opt/intel/inteloneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling DPCPP_Essentials Module7 -- oneDPL Intro sample - 1 of 5 dpl_simple.cpp
dpcpp lab/dpl_simple.cpp -o dpl_simple -w
./dpl_simple
