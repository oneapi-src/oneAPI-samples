#!/bin/bash


export DPCPP_ROOT=$PSCRATCH/llvm-build/install
export NV_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/22.7/cuda/11.7
export PATH=$DPCPP_ROOT/bin:$PATH
export LD_LIBRARY_PATH=$DPCPP_ROOT/lib:$NV_HOME/lib64:$LD_LIBRARY_PATH
rm -rf build
mkdir build
cd build
cmake .. -DCUDA=1 -DNV_HOME=$NV_HOME
make

## Need to run this under sbatch or salloc with your credentials
#echo "Running on gpu"
#ONEAPI_DEVICE_SELECTOR=cuda:0 ./vector-addition-examples
echo "Expected: Sum: 63661.5; Sum neg: -13185.3; Sum pos: 76847.3; checksum: -0.432617"
