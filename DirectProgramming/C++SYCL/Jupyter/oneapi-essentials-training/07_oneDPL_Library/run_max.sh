#!/bin/bash
source /opt/intel/inteloneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling SYCL_Essentials Module7 -- oneDPL Extension APIs - 10 of 12 maximum_function.cpp
icpx -fsycl lab/maximum_function.cpp -w -o bin/maximum_function
bin/maximum_function
