#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling AI numba-dppy Essentials Module2 -- dpctl Intro sample - 1 of 3 simple_dpctl_queue.py
python lab/simple_dpctl_queue.py
