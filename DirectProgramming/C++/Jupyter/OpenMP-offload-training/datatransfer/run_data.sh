#!/bin/bash
/bin/echo "##" $(whoami) is running OMP_Offload Module2 -- Data Transfer - 2 of 2 main_data_region.cpp/f90
echo "########## Executing"
cd bin
./data_region.out
echo "########## Done"
