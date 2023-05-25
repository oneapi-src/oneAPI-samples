#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
icpx -fsycl lab/wg_reduction.cpp 
if [ $? -eq 0 ]; then ./a.out; fi
