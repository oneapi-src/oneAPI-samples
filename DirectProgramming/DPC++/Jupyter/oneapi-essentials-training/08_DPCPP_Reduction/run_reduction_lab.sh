#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling DPCPP_Essentials Module8 -- DPCPP Reduction - 8 of 8 reduction_lab.cpp
dpcpp lab/reduction_lab.cpp 
if [ $? -eq 0 ]; then ./a.out; fi


