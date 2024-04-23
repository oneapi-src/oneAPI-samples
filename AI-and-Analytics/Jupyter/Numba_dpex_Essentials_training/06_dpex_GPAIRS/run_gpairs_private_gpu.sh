#!/bin/bash
source /glob/development-tools/versions/oneapi/2022.2/inteloneapi/setvars.sh --force > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling AI numba-dpex essentials Module6 gpairs - 2 of 5 gpairs_gpu_private_memory.py
python -Wignore lab/gpairs_gpu_private_memory.py --steps 5 --size 1024 --repeat 5 --json result_gpu.json
