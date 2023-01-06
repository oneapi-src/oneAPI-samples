#!/bin/bash
source /opt/intel/inteloneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling SYCL_Essentials Module7 -- oneDPL Extension APIs - 1 of 12 reduce_segment.cpp
icpx -fsycl lab/reduce_segment.cpp -w -o bin/reduce_segment
bin/reduce_segment
