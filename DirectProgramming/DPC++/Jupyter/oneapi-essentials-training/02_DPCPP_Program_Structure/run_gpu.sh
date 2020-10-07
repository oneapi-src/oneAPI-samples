#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling DPCPP_Essentials Module2 -- DPCPP Program Structure sample - 1 of 6 gpu_sample.cpp
dpcpp lab/gpu_sample.cpp -o bin/gpu_sample
if [ $? -eq 0 ]; then bin/gpu_sample; fi

