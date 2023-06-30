default: all

all: sgemm.exe dgemm.exe 

run: sgemm.exe dgemm.exe
	./sgemm.exe
	./dgemm.exe

MKL_COPTS = -DMKL_ILP64  -I"${MKLROOT}/include"
MKL_LIBS = -L${MKLROOT}/lib/intel64 -lmkl_sycl -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lsycl -lOpenCL -lpthread -lm -ldl

DPCPP_OPTS = $(MKL_COPTS) -fsycl-device-code-split=per_kernel $(MKL_LIBS)

sgemm.exe: matrix_mul_mkl.cpp
	icpx -fsycl $< -o $@ $(DPCPP_OPTS)

dgemm.exe: matrix_mul_mkl.cpp
	icpx -fsycl $< -o $@ $(DPCPP_OPTS) -DUSE_DOUBLE

clean:
	del /q sgemm.exe dgemm.exe
