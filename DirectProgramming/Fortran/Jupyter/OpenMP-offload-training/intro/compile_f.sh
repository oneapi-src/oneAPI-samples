#!/bin/bash
source /opt/intel/inteloneapi/setvars.sh > /dev/null 2>&1
echo "########## Compiling"
ifx -fiopenmp -fopenmp-targets=spir64 lab/simple.f90 -o bin/simple
echo "########## Done"
