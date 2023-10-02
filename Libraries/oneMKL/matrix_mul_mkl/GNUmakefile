# Makefile for GNU Make

default: all

all: sgemm.mkl dgemm.mkl

run: sgemm.mkl dgemm.mkl
	./sgemm.mkl
	./dgemm.mkl

INCLUDE_COMMON=../../../common
MKL_COPTS = -DMKL_ILP64  -qmkl=sequential
MKL_LIBS  = -lsycl -lOpenCL -lpthread -lm -ldl

DPCPP_OPTS = -O3 $(MKL_COPTS) $(MKL_LIBS)

sgemm.mkl: matrix_mul_mkl.cpp
	icpx -fsycl -I$(INCLUDE_COMMON) $< -o $@ $(DPCPP_OPTS)

dgemm.mkl: matrix_mul_mkl.cpp
	icpx -fsycl -I$(INCLUDE_COMMON) $< -o $@ $(DPCPP_OPTS) -DUSE_DOUBLE

clean:
	-rm -f sgemm.mkl dgemm.mkl

.PHONY: clean run all
