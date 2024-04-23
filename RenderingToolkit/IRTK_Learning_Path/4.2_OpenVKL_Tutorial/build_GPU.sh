#!/bin/bash
source /opt/intel/oneapi/setvars.sh --force

/bin/echo "##" $(whoami) is building vklTutorialGPU
[ ! -d build ] && mkdir -p build
cd build
rm -rf *
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=icpx ../script_GPU
cmake --build . 

