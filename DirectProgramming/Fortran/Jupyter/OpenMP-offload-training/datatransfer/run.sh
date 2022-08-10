#!/bin/bash
/bin/echo "##" $(whoami) is running OMP_Offload Module2 -- Data Transfer - 1 of 2 main.cpp/main.f90
echo "########## Compiling"
ifx -fiopenmp -fopenmp-targets=spir64 main.f90 -o bin/a.out
echo "########## Executing"
cd bin
./a.out
echo "########## Done"
