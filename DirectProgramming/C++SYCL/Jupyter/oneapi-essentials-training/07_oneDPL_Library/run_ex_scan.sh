#!/bin/bash
source /opt/intel/inteloneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling SYCL_Essentials Module7 -- oneDPL Extension APIs - 3 of 12 exclusive_scan.cpp
icpx -fsycl lab/exclusive_scan.cpp -std=c++17 -w -o bin/exclusive_scan
bin/exclusive_scan
