#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
icpx -fsycl lab/atomics_local.cpp 
if [ $? -eq 0 ]; then ./a.out; fi
