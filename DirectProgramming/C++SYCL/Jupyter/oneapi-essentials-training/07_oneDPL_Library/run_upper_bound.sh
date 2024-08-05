#!/bin/bash
source /opt/intel/inteloneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling SYCL_Essentials Module7 -- oneDPL Extension APIs - 6 of 12 upper_bound.cpp
icpx -fsycl lab/upper_bound.cpp -w -o bin/upper_bound
bin/upper_bound
