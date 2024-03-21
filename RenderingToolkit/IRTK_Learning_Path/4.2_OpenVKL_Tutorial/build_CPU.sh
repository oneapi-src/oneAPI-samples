#!/bin/bash
source /opt/intel/oneapi/setvars.sh --force

/bin/echo "##" $(whoami) is building vklTutorialCPU
[ ! -d build ] && mkdir -p build
cd build
rm -rf *
cmake ../script_CPU
cmake --build . 

