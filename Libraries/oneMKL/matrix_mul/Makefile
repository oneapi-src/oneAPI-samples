DPCPP_CXX = dpcpp
DPCPP_CXXFLAGS = -g -o

MKL_CXXFLAGS = -I$(MKLROOT)/include -DMKL_ILP64
MKL_LDFLAGS = ${MKLROOT}/lib/intel64/libmkl_sycl.a  -L${MKLROOT}/lib/intel64 -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lOpenCL -ldl
MKL_EXE_NAME = matrix_mul_mkl
MKL_SOURCES = src/matrix_mul_mkl.cpp

all:
	$(DPCPP_CXX) $(MKL_CXXFLAGS) $(DPCPP_CXXFLAGS) $(MKL_EXE_NAME) $(MKL_SOURCES) $(MKL_LDFLAGS)

build_mkl:
	$(DPCPP_CXX) $(MKL_CXXFLAGS) $(DPCPP_CXXFLAGS) $(MKL_EXE_NAME) $(MKL_SOURCES) $(MKL_LDFLAGS)


run:
	SYCL_BE=PI_OPENCL ./$(MKL_EXE_NAME)

run_mkl:
	SYCL_BE=PI_OPENCL ./$(MKL_EXE_NAME)


clean: 
	rm -rf $(MKL_EXE_NAME)



