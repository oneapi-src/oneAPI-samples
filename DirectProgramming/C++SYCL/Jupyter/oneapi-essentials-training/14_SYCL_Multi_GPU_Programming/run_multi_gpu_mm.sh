#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
icpx -fsycl -ltbb lab/multi_gpu_mm.cpp -w
unset ONEAPI_DEVICE_SELECTOR
if [ $? -eq 0 ]; then ./a.out; fi

