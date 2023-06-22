#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling AI numba-dpex essentials Module3 -- Pair_wise - 1 of 5 pairwise_distance.py
python -Wignore lab/pairwise_distance.py --steps 5 --size 1024 --repeat 5 --json result_gpu.json
