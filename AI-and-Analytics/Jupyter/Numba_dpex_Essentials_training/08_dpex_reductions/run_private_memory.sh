#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling Reductions --  - 4 of 5 private_memory_kernel.py
python -W ignore lab/private_memory_kernel.py
