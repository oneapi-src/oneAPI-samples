#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling AI numba-dppy essentials Module1 -- DPPY Intro sample - 3 of 4 dbscan.py
python lab/dbscan.py
