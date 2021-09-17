#!/bin/bash
source /opt/intel/inteloneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling DPCPP_Essentials Module7 -- oneDPL Extension APIs - 9 of 12 minimum_function.cpp
dpcpp lab/minimum_function.cpp -w -o bin/minimum_function
bin/minimum_function
