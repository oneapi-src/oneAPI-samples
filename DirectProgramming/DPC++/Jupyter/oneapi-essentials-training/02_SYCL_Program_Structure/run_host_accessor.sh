#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling SYCL_Essentials Module2 -- SYCL Program Structure sample - 3 of 7 host_accessor_sample.cpp
icpx -fsycl lab/host_accessor_sample.cpp
if [ $? -eq 0 ]; then ./a.out; fi

