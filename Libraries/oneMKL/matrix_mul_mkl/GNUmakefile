# Makefile for GNU Make

default: all

all: sgemm.mkl dgemm.mkl

run: sgemm.mkl dgemm.mkl
	./sgemm.mkl
	./dgemm.mkl

MKL_COPTS = -DMKL_ILP64  -I"${MKLROOT}/include"
MKL_LIBS = -L${MKLROOT}/lib -lmkl_sycl -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lsycl -lOpenCL -lpthread -lm -ldl

DPCPP_OPTS = -O3 $(MKL_COPTS) $(MKL_LIBS)

sgemm.mkl: matrix_mul_mkl.cpp
	icpx -fsycl $< -o $@ $(DPCPP_OPTS)

dgemm.mkl: matrix_mul_mkl.cpp
	icpx -fsycl $< -o $@ $(DPCPP_OPTS) -DUSE_DOUBLE

clean:
	-rm -f sgemm.mkl dgemm.mkl

.PHONY: clean run all
