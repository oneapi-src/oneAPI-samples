#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling SYCL_Essentials Module2 -- SYCL Program Structure sample - 7 of 7 vector_add.cpp
icpx -fsycl lab/vector_add.cpp
if [ $? -eq 0 ]; then ./a.out; fi

