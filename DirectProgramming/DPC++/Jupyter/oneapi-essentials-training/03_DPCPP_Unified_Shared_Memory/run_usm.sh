#!/bin/bash
source /opt/intel/inteloneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling DPCPP_Essentials Module3 -- DPCPP Unified Shared Memory - 1 of 4 usm.cpp
dpcpp lab/usm.cpp 
./a.out
