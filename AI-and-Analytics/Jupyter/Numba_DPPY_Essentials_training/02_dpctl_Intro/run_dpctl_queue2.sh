#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling AI numba-dppy essentials Module2 -- dpctl Intro sample - 1 of 3 dpctl_queue2.py
python lab/dpctl_queue_2.py

