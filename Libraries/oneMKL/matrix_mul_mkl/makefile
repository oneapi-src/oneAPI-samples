# Makefile for NMAKE

default: run

all: matrix_mul_mkl.exe

run: matrix_mul_mkl.exe
	.\matrix_mul_mkl.exe single
	.\matrix_mul_mkl.exe double

DPCPP_OPTS=/I"$(MKLROOT)\include" /Qmkl /Qmkl-sycl-impl=blas /EHsc -fsycl-device-code-split=per_kernel OpenCL.lib

matrix_mul_mkl.exe: matrix_mul_mkl.cpp
	icx-cl -fsycl matrix_mul_mkl.cpp /Fematrix_mul_mkl.exe $(DPCPP_OPTS)

clean:
	del /q matrix_mul_mkl.exe matrix_mul_mkl.exp matrix_mul_mkl.lib

pseudo: clean run all
