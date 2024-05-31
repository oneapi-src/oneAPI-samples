#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
icpx -fsycl -ltbb lab/multi_gpu_vadd_split.cpp -w

export ONEAPI_DEVICE_SELECTOR="level_zero:0;level_zero:1"

if [ $? -eq 0 ]; then ./a.out; fi
