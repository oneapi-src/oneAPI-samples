#!/bin/bash
source /opt/intel/inteloneapi/setvars.sh > /dev/null 2>&1

#Command Line Arguments
arg=" -n 1024" # set matrix size
src="lab/"

echo ====================
echo mm_dpcpp_basic
dpcpp ${src}mm_dpcpp_basic.cpp ${src}mm_dpcpp_common.cpp -o ${src}mm_dpcpp_basic -w -O3
./${src}mm_dpcpp_basic$arg


