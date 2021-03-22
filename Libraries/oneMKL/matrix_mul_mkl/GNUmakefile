# Makefile for GNU Make

default: run

all: run

run: matrix_mul_mkl
	./matrix_mul_mkl

MKL_COPTS = -DMKL_ILP64  -I"${MKLROOT}/include"
MKL_LIBS = -L${MKLROOT}/lib/intel64 -lmkl_sycl -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lsycl -lOpenCL -lpthread -lm -ldl

DPCPP_OPTS = $(MKL_COPTS) -fsycl-device-code-split=per_kernel $(MKL_LIBS)

matrix_mul_mkl: matrix_mul_mkl.cpp
	dpcpp $< -o $@ $(DPCPP_OPTS)

clean:
	-rm -f matrix_mul_mkl

.PHONY: clean run all
