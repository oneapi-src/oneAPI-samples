#!/bin/bash
source /opt/intel/inteloneapi/setvars.sh > /dev/null 2>&1
echo "########## Compiling"
icpx -fiopenmp -fopenmp-targets=spir64 main_data_region.cpp -o bin/data_region.out
echo "########## Done"
