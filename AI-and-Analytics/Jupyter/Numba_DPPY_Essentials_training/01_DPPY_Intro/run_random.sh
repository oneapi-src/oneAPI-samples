#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling AI essentials Module1 -- DPPY Intro sample - 1 of 2 numba_dppy_random.py
python lab/numba_dppy_random.py
