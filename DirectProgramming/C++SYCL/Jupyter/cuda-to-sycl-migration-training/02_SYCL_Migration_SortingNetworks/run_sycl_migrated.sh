#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
cd sycl_migrated/Samples/2_Concepts_and_Techniques/sortingNetworks
icpx -fsycl -I ../../../Common -I ../../../include *.cpp
if [ $? -eq 0 ]; then ./a.out; fi

