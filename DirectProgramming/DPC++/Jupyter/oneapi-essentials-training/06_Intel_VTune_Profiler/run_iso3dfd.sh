#!/bin/bash
source /opt/intel/inteloneapi/setvars.sh > /dev/null 2>&1
dpcpp src/iso3dfd.cpp src/utils.cpp src/iso3dfd_kernels.cpp -o iso3dfd
./iso3dfd 256 256 256 8 8 8 20 sycl gpu
