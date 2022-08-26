#!/bin/bash

#PBS -l nodes=1:fpga_compile:ppn=2
#PBS -d .

source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1

echo
echo start: $(date "+%y/%m/%d %H:%M:%S.%3N")
echo

make fpga_emu -f Makefile.fpga

echo
echo stop: $(date "+%y/%m/%d %H:%M:%S.%3N")
echo
