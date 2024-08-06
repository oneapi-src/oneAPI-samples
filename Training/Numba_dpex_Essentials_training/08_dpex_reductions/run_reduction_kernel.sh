#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling Reductions --  - 1 of 5 reduction_kernel.py
python -W ignore lab/reduction_kernel.py
