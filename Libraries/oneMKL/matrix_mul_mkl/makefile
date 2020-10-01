# Makefile for NMAKE

default: run

run: matrix_mul_mkl.exe
	.\matrix_mul_mkl.exe

DPCPP_OPTS=/I"$(MKLROOT)\include" /Qmkl /EHsc -fsycl-device-code-split=per_kernel OpenCL.lib

matrix_mul_mkl.exe: matrix_mul_mkl.cpp
	dpcpp matrix_mul_mkl.cpp /Fematrix_mul_mkl.exe $(DPCPP_OPTS)

clean:
	del /q matrix_mul_mkl.exe matrix_mul_mkl.exp matrix_mul_mkl.lib

pseudo: clean run
