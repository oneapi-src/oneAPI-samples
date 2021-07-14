#!/bin/bash
source /opt/intel/inteloneapi/setvars.sh > /dev/null 2>&1
echo "########## Compiling"
ifx -fiopenmp -fopenmp-targets=spir64 main_data_region.f90 -o bin/data_region.out
echo "########## Done"
