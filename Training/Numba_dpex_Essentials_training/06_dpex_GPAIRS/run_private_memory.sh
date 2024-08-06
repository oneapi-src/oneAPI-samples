#!/bin/bash
source /glob/development-tools/versions/oneapi/2022.2/inteloneapi/setvars.sh --force > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling Private memory -- Gpairs - 1 of 5 private_memory_kernel.py
python -Wignore lab/private_memory_kernel.py
