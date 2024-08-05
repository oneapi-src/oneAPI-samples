#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
icpx -fsycl lab/convolution_global_conditionals_minmax.cpp 
if [ $? -eq 0 ]; then ./a.out; fi
