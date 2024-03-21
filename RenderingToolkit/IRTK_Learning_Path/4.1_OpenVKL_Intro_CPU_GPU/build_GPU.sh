#!/bin/bash
source /opt/intel/oneapi/setvars.sh --force
rm -r build_GPU
mkdir build_GPU
cd build_GPU
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=icpx ../script_GPU
cmake --build . --verbose
