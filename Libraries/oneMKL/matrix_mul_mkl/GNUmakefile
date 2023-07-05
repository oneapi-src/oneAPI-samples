# Makefile for GNU Make

default: all

all: sgemm.mkl dgemm.mkl

run: sgemm.mkl dgemm.mkl
	./sgemm.mkl
	./dgemm.mkl

MKL_COPTS = 
MKL_LIBS = -qmkl

DPCPP_OPTS = -O3 $(MKL_COPTS) $(MKL_LIBS)

sgemm.mkl: matrix_mul_mkl.cpp
	icpx -fsycl $< -o $@ $(DPCPP_OPTS)

dgemm.mkl: matrix_mul_mkl.cpp
	icpx -fsycl $< -o $@ $(DPCPP_OPTS) -DUSE_DOUBLE

clean:
	-rm -f sgemm.mkl dgemm.mkl

.PHONY: clean run all
