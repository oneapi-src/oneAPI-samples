#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling AI numba-dpex essentials Module4 --  Black scholess - 4 of 4 black_sholes_numpy.py
python -Wignore lab/black_sholes_numpy_graph.py --steps 5 --size 1024 --repeat 1 --json result_gpu.json
