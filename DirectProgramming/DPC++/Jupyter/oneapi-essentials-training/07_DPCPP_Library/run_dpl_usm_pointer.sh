#!/bin/bash
source /opt/intel/inteloneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling DPCPP_Essentials Module7 -- oneDPL Intro sample - 4 of 5 dpl_usm_pointer.cpp
dpcpp lab/dpl_usm_pointer.cpp -o dpl_usm_pointer -w
./dpl_usm_pointer
