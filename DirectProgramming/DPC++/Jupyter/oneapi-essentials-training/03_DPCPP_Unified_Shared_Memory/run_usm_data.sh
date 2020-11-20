#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling DPCPP_Essentials Module3 -- DPCPP Unified Shared Memory - 3 of 4 usm_data.cpp
dpcpp lab/usm_data.cpp 
if [ $? -eq 0 ]; then ./a.out; fi

