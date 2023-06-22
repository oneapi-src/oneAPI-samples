#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling AI numba-dpex essentials Module3 -- Gpairs - 1 of 5 gpairs.py
python -Wignore lab/gpairs.py --steps 5 --size 1024 --repeat 5 --json result.json
