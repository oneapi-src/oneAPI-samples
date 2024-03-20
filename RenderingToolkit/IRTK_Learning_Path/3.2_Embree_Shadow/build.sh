#!/bin/bash

source /opt/intel/oneapi/setvars.sh

[ ! -d build ] && mkdir -p build
cd build

rm -rf *
cmake ..
cmake --build .