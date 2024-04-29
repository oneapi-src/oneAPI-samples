#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
cd sycl_migrated_option2/Samples/3_CUDA_Features/simpleCudaGraphs
if [ ! -d bin ]; then mkdir bin; fi
icpx -fsycl -fsycl-targets=intel_gpu_pvc -I ../../../Common -I ../../../include *.cpp -pthread -w
clear
if [ $? -eq 0 ]; then ./a.out; fi


