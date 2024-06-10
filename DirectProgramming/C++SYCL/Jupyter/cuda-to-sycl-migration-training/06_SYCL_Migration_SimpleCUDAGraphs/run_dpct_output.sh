#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
cd dpct_output/Samples/3_CUDA_Features/simpleCudaGraphs
icpx -fsycl -fsycl-targets=intel_gpu_pvc -I ../../../Common -I ../../../include *.cpp -w
clear
if [ $? -eq 0 ]; then ./a.out; fi

