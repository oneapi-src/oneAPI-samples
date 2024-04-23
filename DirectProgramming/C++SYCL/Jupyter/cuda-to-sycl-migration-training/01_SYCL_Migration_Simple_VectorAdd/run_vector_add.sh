#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
icpx -fsycl -I include sycl_migrated/vectoradd.dp.cpp
if [ $? -eq 0 ]; then ./a.out; fi
