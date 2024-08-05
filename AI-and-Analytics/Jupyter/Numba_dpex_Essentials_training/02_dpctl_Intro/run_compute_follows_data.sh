#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling AI numba-dpex essentials Module1 -- DPPY Intro sample - 3 of 4 compute_follows_data.py
python -Wignore lab/compute_follows_data.py
