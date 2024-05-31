#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
icpx -fsycl -ltbb lab/offload_mm.cpp -w -o offload_mm.out
if [ $? -eq 0 ]; then echo "Compile Complete"; fi

