#!/bin/bash
source /opt/intel/inteloneapi/setvars.sh > /dev/null 2>&1

#Command Line Arguments
arg=" -n 1024" # set matrix  size
src="lab/"

echo mm_dpcpp_mkl
dpcpp ${src}mm_dpcpp_mkl.cpp ${src}mm_dpcpp_common.cpp -DMKL_ILP64 -I$MKLROOT/include -L$MKLROOT/lib/intel64 -lmkl_sycl -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lsycl -lOpenCL -lpthread -lm -ldl -O3 -o ${src}mm_dpcpp_mkl
./${src}mm_dpcpp_mkl$arg
