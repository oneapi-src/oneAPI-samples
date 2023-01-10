#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling SYCL_Essentials Module2 -- SYCL Program Structure sample - 4 of 7 buffer_destruction2.cpp
icpx -fsycl lab/buffer_destruction2.cpp
if [ $? -eq 0 ]; then ./a.out; fi

