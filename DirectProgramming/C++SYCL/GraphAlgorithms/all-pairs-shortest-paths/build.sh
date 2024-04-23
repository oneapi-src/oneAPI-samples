#!/bin/bash

#PBS -l nodes=1:gpu:ppn=2
#PBS -d .

source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1

echo
echo start: $(date "+%y/%m/%d %H:%M:%S.%3N")
echo

mkdir build
cd build
cmake ..
make all

echo
echo stop: $(date "+%y/%m/%d %H:%M:%S.%3N")
echo
