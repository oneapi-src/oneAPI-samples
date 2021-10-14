#!/bin/bash
source /opt/intel/inteloneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling DPCPP_Essentials Module7 -- DPC++ Libraries - 12 of 13 ranges.cpp
dpcpp lab/ranges.cpp -std=c++17 -o bin/ranges
bin/ranges
