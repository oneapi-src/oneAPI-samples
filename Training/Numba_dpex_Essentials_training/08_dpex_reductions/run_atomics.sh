#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling Reductions --  - 3 of 5 atomics_kernel.py
python -W ignore lab/atomics_kernel.py
