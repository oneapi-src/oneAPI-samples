#!/bin/bash
source /opt/intel/inteloneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling DPCPP_Essentials Module7 -- oneDPL Extension APIs - 2 of 12 inclusive_scan.cpp
dpcpp lab/inclusive_scan.cpp -w -o bin/inclusive_scan
bin/inclusive_scan
