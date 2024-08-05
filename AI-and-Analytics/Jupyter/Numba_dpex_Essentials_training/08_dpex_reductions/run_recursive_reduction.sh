#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling Reductions --  - 5 of 5 recursive_reduction_kernel.py
python -W ignore lab/recursive_reduction_kernel.py
