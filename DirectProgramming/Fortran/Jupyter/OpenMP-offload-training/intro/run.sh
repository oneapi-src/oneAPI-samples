#!/bin/bash
/bin/echo "##" $(whoami) is running OMP_Offload Module1 -- Intro to OpenMP offload - 1 of 1 simple.cpp/f90
echo "########## Compiling"
ifx -qopenmp -fopenmp-targets=spir64 lab/simple.f90 -o bin/simple
echo "########## Executing"
cd bin
./simple
echo "########## Done"
