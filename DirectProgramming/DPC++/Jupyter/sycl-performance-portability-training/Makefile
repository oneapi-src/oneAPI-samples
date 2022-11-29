all: mm_dpcpp_mkl mm_dpcpp_basic mm_dpcpp_basic_full mm_dpcpp_ndrange mm_dpcpp_ndrange_var mm_dpcpp_localmem mm_dpcpp_localmem_wg

mm_dpcpp_mkl: lab/mm_dpcpp_mkl.cpp lab/mm_dpcpp_common.cpp
	icpx lab/mm_dpcpp_mkl.cpp lab/mm_dpcpp_common.cpp -DMKL_ILP64 -I$MKLROOT/include -L$MKLROOT/lib/intel64 -fsycl -lmkl_sycl -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lsycl -lOpenCL -lpthread -lm -ldl -O3 -o lab/mm_dpcpp_mkl

mm_dpcpp_basic: lab/mm_dpcpp_basic.cpp lab/mm_dpcpp_common.cpp
	icpx -fsycl lab/mm_dpcpp_basic.cpp lab/mm_dpcpp_common.cpp -o lab/mm_dpcpp_basic

mm_dpcpp_basic_full: lab/mm_dpcpp_basic_full.cpp
	icpx -fsycl lab/mm_dpcpp_basic_full.cpp -o lab/mm_dpcpp_basic_full

mm_dpcpp_ndrange: lab/mm_dpcpp_ndrange.cpp lab/mm_dpcpp_common.cpp
	icpx -fsycl lab/mm_dpcpp_ndrange.cpp lab/mm_dpcpp_common.cpp -o lab/mm_dpcpp_ndrange

mm_dpcpp_ndrange_var: lab/mm_dpcpp_ndrange_var.cpp lab/mm_dpcpp_common.cpp
	icpx -fsycl lab/mm_dpcpp_ndrange_var.cpp lab/mm_dpcpp_common.cpp -o lab/mm_dpcpp_ndrange_var

mm_dpcpp_localmem: lab/mm_dpcpp_localmem.cpp lab/mm_dpcpp_common.cpp
	icpx -fsycl lab/mm_dpcpp_localmem.cpp lab/mm_dpcpp_common.cpp -o lab/mm_dpcpp_localmem

mm_dpcpp_localmem_wg: lab/mm_dpcpp_localmem.cpp lab/mm_dpcpp_common_wg.cpp
	icpx -fsycl lab/mm_dpcpp_localmem.cpp lab/mm_dpcpp_common_wg.cpp -o lab/mm_dpcpp_localmem_wg

run:
	lab/mm_dpcpp_mkl
	lab/mm_dpcpp_basic
	lab/mm_dpcpp_basic_full
	lab/mm_dpcpp_ndrange
	lab/mm_dpcpp_ndrange_var
	lab/mm_dpcpp_localmem
	lab/mm_dpcpp_localmem_wg

clean:
	rm -rf lab/mm_dpcpp_mkl lab/mm_dpcpp_basic lab/mm_dpcpp_basic_full lab/mm_dpcpp_ndrange lab/mm_dpcpp_ndrange_var lab/mm_dpcpp_localmem lab/mm_dpcpp_localmem_wg
