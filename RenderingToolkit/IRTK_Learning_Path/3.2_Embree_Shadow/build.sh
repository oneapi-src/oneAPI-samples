#!/bin/bash

source /opt/intel/oneapi/setvars.sh --force

[ ! -d build ] && mkdir -p build
cd build

rm -rf *
cmake ..
cmake --build .