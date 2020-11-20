#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling DPCPP_Essentials Module7 -- oneDPL Intro sample - 2 of 5 dpl_sortdouble.cpp
dpcpp lab/dpl_sortdouble.cpp -o dpl_sortdouble
if [ $? -eq 0 ]; then ./dpl_sortdouble; fi

