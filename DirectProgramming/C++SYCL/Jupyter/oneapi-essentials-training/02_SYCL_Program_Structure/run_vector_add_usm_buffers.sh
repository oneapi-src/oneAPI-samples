#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling DPCPP_Essentials Module2 -- SYCL Program Structure sample - 8 of 8 vector_add_usm_buffers.cpp
icpx -fsycl lab/vector_add_usm_buffers.cpp
if [ $? -eq 0 ]; then ./a.out; fi

