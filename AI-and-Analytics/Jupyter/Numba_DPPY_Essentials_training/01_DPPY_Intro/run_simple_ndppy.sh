#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling AI essentials Module1 -- DPPY Intro sample - 3 of 8 simple_numba_dppy.py
python lab/simple_numba_dppy.py
