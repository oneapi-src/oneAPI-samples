#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling SYCL_Essentials Module2 -- SYCL Program Structure sample - 6 of 7 complex_mult.cpp
#icpx -fsycl complex_mult.cpp -o bin/complex_mult -I ./include
#bin/complex_mult
chmod 755 q
make clean
make all
make run

