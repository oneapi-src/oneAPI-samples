#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling AI essentials Module1 -- DPPY Pair Wise - 5 of 5 pair_wise_numpy.py
python lab/pair_wise_numpy.py --steps 5 --size 1024 --repeat 5 --json result_gpu.json
