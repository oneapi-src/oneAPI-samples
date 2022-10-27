#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is running OMP_Offload Module2 -- Data Transfer - 2 of 2 main_data_region.cpp/f90
echo "########## Compiling"
icpx -qopenmp -fopenmp-targets=spir64 main_data_region.cpp -o bin/data_region.out || exit $?
echo "########## Executing"
cd bin
./data_region.out
echo "########## Done"
