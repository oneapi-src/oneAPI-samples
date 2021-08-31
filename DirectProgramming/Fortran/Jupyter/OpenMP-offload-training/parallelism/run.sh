#!/bin/bash
/bin/echo "##" $(whoami) is running OMP_Offload Module3 -- Parallelism - 1 of 1 main.cpp/main.f90
echo "########## Executing"
cd bin
./a.out
echo "########## Done"
