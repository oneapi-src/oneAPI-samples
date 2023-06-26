#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
icpx -fsycl lab/usm_overlap_copy.cpp 
if [ $? -eq 0 ]; then ./a.out; fi
