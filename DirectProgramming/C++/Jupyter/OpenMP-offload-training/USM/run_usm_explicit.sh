#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is running OMP_Offload Module4 2/2 -- USM usm_explicit.cpp/usm_explicit.f90
source /opt/intel/inteloneapi/setvars.sh > /dev/null 2>&1
echo "########## Compiling"
icpx -qopenmp -fopenmp-targets=spir64 lab/usm_explicit.cpp -o bin/a.out || exit $?
echo "########## Executing"
cd bin
./a.out
echo "########## Done"

