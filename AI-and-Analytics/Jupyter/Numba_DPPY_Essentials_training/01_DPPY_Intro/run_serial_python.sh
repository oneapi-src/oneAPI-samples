#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling AI numba-dppy essentials Module1 -- DPPY Intro sample - 1 of 4 serial_python.py
python lab/serial_python.py
