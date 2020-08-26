#!/bin/bash
source /opt/intel/inteloneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling DPCPP_Essentials Module7 -- oneDPL Intro sample - 2 of 5 dpl_sortdouble.cpp
dpcpp lab/dpl_sortdouble.cpp -o dpl_sortdouble -w
./dpl_sortdouble
