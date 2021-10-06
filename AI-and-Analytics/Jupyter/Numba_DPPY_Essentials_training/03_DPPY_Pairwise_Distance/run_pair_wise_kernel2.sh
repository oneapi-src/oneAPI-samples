#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling AI numba-dppy essentials Module3--  Pair wise - 4 of 5 pair_wise_kernel2.py
python lab/pair_wise_kernel2.py
