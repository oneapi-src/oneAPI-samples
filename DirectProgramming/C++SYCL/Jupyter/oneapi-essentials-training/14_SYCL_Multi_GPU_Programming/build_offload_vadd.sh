#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
icpx -fsycl -ltbb lab/multi_gpu_vadd_split.cpp -w -o offload_vadd.out
if [ $? -eq 0 ]; then echo "Compile Complete"; fi

