#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling AI essentials Module1 -- DPPY Black sholes - 1 of 2 black_sholes_jit.py
python lab/black_sholes_jit.py --steps 5 --size 1024 --repeat 5
