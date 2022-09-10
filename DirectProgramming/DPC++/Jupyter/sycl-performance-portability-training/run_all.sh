#!/bin/bash
source /opt/intel/inteloneapi/setvars.sh > /dev/null 2>&1

#Command Line Arguments
arg=" -n 1024" # set matrix  size
#arg=" -n 32 -m 16 -p -v" # set matrix size, work-group, print output, verify result
#arg=" -n 256 -v" # set matrix size, verify output
src="lab/"
common="mm_dpcpp_common.cpp"

#echo ====================
#echo mm_dpcpp_basic
#dpcpp ${src}mm_dpcpp_basic.cpp ${src}mm_dpcpp_common.cpp -o ${src}mm_dpcpp_basic -w -O3
#./${src}mm_dpcpp_basic$arg

echo ====================
echo mm_dpcpp_ndrange
dpcpp ${src}mm_dpcpp_ndrange.cpp ${src}${common} -o ${src}mm_dpcpp_ndrange -w -O3
./${src}mm_dpcpp_ndrange$arg

echo ====================
echo mm_dpcpp_ndrange_var
dpcpp ${src}mm_dpcpp_ndrange_var.cpp ${src}${common} -o ${src}mm_dpcpp_ndrange_var -w -O3
./${src}mm_dpcpp_ndrange_var$arg

echo ====================
echo mm_dpcpp_localmem
dpcpp ${src}mm_dpcpp_localmem.cpp ${src}${common} -o ${src}mm_dpcpp_localmem -w -O3
./${src}mm_dpcpp_localmem$arg

echo ====================
echo mm_dpcpp_mkl
dpcpp ${src}mm_dpcpp_mkl.cpp ${src}${common} -DMKL_ILP64 -I$MKLROOT/include -L$MKLROOT/lib/intel64 -lmkl_sycl -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lsycl -lOpenCL -lpthread -lm -ldl -O3 -o ${src}mm_dpcpp_mkl
./${src}mm_dpcpp_mkl$arg
