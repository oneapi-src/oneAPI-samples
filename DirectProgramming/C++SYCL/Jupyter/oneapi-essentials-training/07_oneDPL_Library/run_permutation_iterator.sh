#!/bin/bash
source /opt/intel/inteloneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling SYCL_Essentials Module7 -- oneDPL Extension APIs - 9 of 12 permutation_iterator.cpp
icpx -fsycl lab/permutation_iterator.cpp -w -o bin/permutation_iterator
bin/permutation_iterator
