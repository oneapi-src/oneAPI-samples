#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling SYCL_Essentials Module9 -- SYCL Advanced Buffers and Accessors sample - 9 of 10 buffer_creation_uncommon.cpp
icpx -fsycl lab/buffer_creation_uncommon.cpp
if [ $? -eq 0 ]; then ./a.out; fi

