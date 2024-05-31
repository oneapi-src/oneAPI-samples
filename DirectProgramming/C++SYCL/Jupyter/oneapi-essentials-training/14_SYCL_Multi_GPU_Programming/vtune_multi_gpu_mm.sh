#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
icpx -fsycl -ltbb lab/multi_gpu_mm.cpp -w
unset ONEAPI_DEVICE_SELECTOR

rm -rf vtune_mm
vtune --collect gpu-hotspots --result-dir vtune_mm $(pwd)/a.out

