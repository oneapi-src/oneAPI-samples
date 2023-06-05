#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling AI numba-dpex Essentials Module2 -- dpctl Intro sample - 3 of 3 dpctl_mem_sample.py
python -Wignore lab/dpctl_mem_sample.py
