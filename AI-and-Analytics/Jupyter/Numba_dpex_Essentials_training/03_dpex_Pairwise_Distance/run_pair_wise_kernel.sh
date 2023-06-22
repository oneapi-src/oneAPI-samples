#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling AI numba-dpex essentials Module3--  Pair wise - 3 of 5 pair_wise_kernel.py
python -Wignore lab/pair_wise_kernel.py --steps 5 --size 1024 --repeat 5 --json result_gpu.json -d 3
