#!/bin/bash
/bin/echo "##" $(whoami) is running OMP_Offload Module2 -- Data Transfer - 2 of 2 main_data_region.cpp/f90
echo "########## Compiling"
ifx -fiopenmp -fopenmp-targets=spir64 main_data_region.f90 -o bin/data_region.out
echo "########## Executing"
cd bin
./data_region.out
echo "########## Done"
