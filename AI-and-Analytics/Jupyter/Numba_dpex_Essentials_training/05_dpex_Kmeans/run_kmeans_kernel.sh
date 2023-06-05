#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling AI numba-dpex essentials Module5 --  K-Means - 2 of 3 kmeans_kernel.py
python -Wignore lab/kmeans_kernel.py --steps 5 --size 1024 --repeat 5 --json result_gpu.json
