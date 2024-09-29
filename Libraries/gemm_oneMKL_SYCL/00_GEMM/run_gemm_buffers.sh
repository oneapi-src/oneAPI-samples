#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling oneMKL_introduction Module0 -- gemm with buffers - 1 of 3 dpcpp_gemm_buffers.cpp

icpx -fsycl -fsycl-device-code-split=per_kernel -DMKL_ILP64 -I$MKLROOT/include -L$MKLROOT/lib/intel64 -lmkl_sycl -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lsycl -lOpenCL -lpthread -lm -ldl lab/dpcpp_gemm_buffers.cpp

if [ $? -eq 0 ]; then ./a.out; fi
