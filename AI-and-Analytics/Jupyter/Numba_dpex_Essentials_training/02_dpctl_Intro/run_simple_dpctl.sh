#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling AI numba-dpex Essentials Module2 -- dpctl Intro sample - 2 of 3 simple_dpctl.py
python -Wignore lab/simple_dpctl.py
