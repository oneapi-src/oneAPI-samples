#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
icpx -fsycl sycl_migrated_optimized/vectoradd.dp.cpp 
if [ $? -eq 0 ]; then ./a.out; fi