#!/bin/bash
source /opt/intel/inteloneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling DPCPP_Essentials Module7 -- oneDPL Extension APIs - 10 of 12 maximum_function.cpp
dpcpp lab/maximum_function.cpp -w -o bin/maximum_function
bin/maximum_function
