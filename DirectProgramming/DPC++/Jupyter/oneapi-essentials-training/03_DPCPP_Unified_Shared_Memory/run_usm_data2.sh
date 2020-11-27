#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling DPCPP_Essentials Module3 -- DPCPP Unified Shared Memory - 4 of 4 usm_data2.cpp
dpcpp lab/usm_data2.cpp 
if [ $? -eq 0 ]; then ./a.out; fi

