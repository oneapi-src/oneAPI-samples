#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling DPCPP_Essentials Module8 -- DPCPP Reduction - 7 of 7 oneapi_reduction_custom_operator.cpp
dpcpp lab/oneapi_reduction_custom_operator.cpp 
if [ $? -eq 0 ]; then ./a.out; fi


