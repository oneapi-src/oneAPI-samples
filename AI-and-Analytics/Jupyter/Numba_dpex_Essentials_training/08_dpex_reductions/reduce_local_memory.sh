#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling Reductions --  - 2 of 5 reduce_local_memory.py
python -W ignore lab/reduce_local_memory.py
