#!/bin/bash

source /opt/intel/oneapi/setvars.sh --force

export rkcommon_DIR=/opt/intel/oneapi/rkcommon/latest/lib/cmake/rkcommon

[ ! -d build_GPU ] && mkdir -p build_GPU
cd build_GPU
rm -rf *
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=icpx -DCMAKE_INSTALL_PREFIX=.. ../script_GPU
cmake --build .