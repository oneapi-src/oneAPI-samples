#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling SYCL_Essentials Module2 -- SYCL Program Structure sample - 5 of 7 custom_device_sample.cpp
icpx -fsycl lab/custom_device_sample.cpp
if [ $? -eq 0 ]; then ./a.out; fi

