#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling AI numba-dpex essentials Module7 --  KNN - 1 of 3 knn_jit.py
python -Wignore lab/knn_jit.py --steps 5 --size 1024 --repeat 5 --json result_gpu.json
