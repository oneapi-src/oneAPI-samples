#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
cd dpct_output/Samples/4_CUDA_Libraries/oceanFFT
if [ ! -d bin ]; then mkdir bin; fi
cp -rf ../../../data/* bin/
icpx -fsycl -fsycl-targets=intel_gpu_pvc -I ../../../Common -I ../../../include *.cpp -qmkl -w
clear
if [ $? -eq 0 ]; then ./a.out qatest; fi

