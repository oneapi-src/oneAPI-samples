#!/bin/bash

#PBS -l nodes=1:gpu:ppn=2
#PBS -d .

#source /opt/intel/oneapi/setvars.sh
rm -rf build
mkdir build
cd build
cmake ..
make

echo "Running on gpu"
ONEAPI_DEVICE_SELECTOR=level_zero:gpu ./vector-addition-examples
echo "Running on cpu"
ONEAPI_DEVICE_SELECTOR=level_zero:cpu ./vector-addition-examples

echo "Expected: Sum: 63661.5; Sum neg: -13185.3; Sum pos: 76847.3; checksum: -0.432617"
