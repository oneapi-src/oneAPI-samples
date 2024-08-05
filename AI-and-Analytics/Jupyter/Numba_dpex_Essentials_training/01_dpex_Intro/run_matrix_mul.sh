#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling AI numba-dpex essentials Module1 -- DPPY Intro sample - 5 of 5 matrix_mul.py
python -Wignore lab/matrix_mul.py
