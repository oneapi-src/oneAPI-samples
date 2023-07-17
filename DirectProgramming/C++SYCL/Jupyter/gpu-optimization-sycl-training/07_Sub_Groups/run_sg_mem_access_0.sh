#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
icpx -fsycl lab/sg_mem_access_0.cpp 
if [ $? -eq 0 ]; then ./a.out; fi
