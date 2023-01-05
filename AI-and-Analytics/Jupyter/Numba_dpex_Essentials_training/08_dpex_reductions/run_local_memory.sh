#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling reductions - 4 of 5 local_memory.py
python -W ignore lab/local_memory_kernel.py
