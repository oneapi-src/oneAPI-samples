#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling AI numba-dpex essentials Module7 --  KNN - 3 of 3 knn_kernel_graph.py
python -Wignore lab/knn_kernel_graph.py --steps 5 --size 1024 --repeat 1 --json result_gpu.json
