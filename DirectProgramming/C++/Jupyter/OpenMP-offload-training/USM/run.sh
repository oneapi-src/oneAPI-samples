#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is running OMP_Offload Module4 1/2 -- USM main.cpp/main.f90
echo "########## Compiling"
icpx -qopenmp -fopenmp-targets=spir64 main.cpp -o bin/a.out || exit $?
echo "########## Executing"
cd bin
./a.out
echo "########## Done"
