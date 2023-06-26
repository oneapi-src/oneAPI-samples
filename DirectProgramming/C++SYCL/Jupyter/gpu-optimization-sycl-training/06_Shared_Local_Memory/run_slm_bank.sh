#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
icpx -fsycl lab/slm_bank.cpp 
if [ $? -eq 0 ]; then ./a.out; fi
