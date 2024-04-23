#!/bin/bash

source /opt/intel/oneapi/setvars.sh --force

[ ! -d build_CPU ] && mkdir -p build_CPU
cd build_CPU
rm -rf *
cmake ../script_CPU
cmake --build .