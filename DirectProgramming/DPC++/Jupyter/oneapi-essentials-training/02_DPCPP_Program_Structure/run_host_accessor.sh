#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling DPCPP_Essentials Module2 -- DPCPP Program Structure sample - 3 of 6 host_accessor_sample.cpp
dpcpp lab/host_accessor_sample.cpp -o bin/host_accessor_sample
if [ $? -eq 0 ]; then bin/host_accessor_sample; fi

