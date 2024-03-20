#!/bin/bash
source /opt/intel/oneapi/setvars.sh
[ -d build ] && rm -rf build
mkdir build
cd build
cmake ..
cmake --build .
