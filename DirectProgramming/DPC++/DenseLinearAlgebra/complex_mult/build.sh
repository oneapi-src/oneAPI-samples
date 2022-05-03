#!/bin/bash

#PBS -l nodes=1:gpu:ppn=2
#PBS -d .

source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1

echo
echo start: $(date "+%y/%m/%d %H:%M:%S.%3N")
echo

make all

make build_usm

echo
echo stop: $(date "+%y/%m/%d %H:%M:%S.%3N")
echo
