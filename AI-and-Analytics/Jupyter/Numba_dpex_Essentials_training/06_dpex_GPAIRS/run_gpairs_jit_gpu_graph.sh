#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling AI numba-dpex essentials Module6 gpairs - 2 of 5 gpairs_gpu_graph.py
python -Wignore lab/gpairs_gpu_graph.py --steps 5 --size 1024 --repeat 1 --json result_gpu.json
