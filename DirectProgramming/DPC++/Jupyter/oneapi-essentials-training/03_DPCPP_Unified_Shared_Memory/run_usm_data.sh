#!/bin/bash
source /opt/intel/inteloneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling DPCPP_Essentials Module3 -- DPCPP Unified Shared Memory - 3 of 4 usm_data.cpp
dpcpp lab/usm_data.cpp 
./a.out
