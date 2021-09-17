#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling DPCPP_Essentials Module7 -- oneDPL Gamma Correction - 1 of 1 gamma_correction.cpp
rm -rf gamma-correction/build
cd gamma-correction &&
mkdir build &&  
cd build &&  
cmake ../. &&  
make
make run

