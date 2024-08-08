#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
cd sycl_migrated/Samples/5_Domain_Specific/MonteCarloMultiGPU
if [ ! -d bin ]; then mkdir bin; fi
icpx -fsycl -fsycl-targets=intel_gpu_pvc -I ../../../Common -I ../../../include *.cpp -qmkl -pthread -w
export ONEAPI_DEVICE_SELECTOR=level_zero:*
clear
if [ $? -eq 0 ]; then ./a.out ; fi

