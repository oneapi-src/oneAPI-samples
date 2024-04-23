#!/bin/bash
source /opt/intel/oneapi/setvars.sh --force
rm -r build_CPU
mkdir build_CPU
cd build_CPU
cmake ../script_CPU
cmake --build . --verbose
