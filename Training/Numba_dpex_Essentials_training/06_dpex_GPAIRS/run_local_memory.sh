#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling Private memory -- Gpairs - 1 of 5 local_memory_kernel.py
python -Wignore lab/local_memory_kernel.py
