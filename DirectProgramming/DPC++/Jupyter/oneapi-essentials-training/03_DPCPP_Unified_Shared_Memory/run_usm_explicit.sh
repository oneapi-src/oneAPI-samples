#!/bin/bash
source /opt/intel/inteloneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling DPCPP_Essentials Module3 -- DPCPP Unified Shared Memory - 2 of 4 usm_explicit.cpp
dpcpp lab/usm_explicit.cpp 
./a.out
