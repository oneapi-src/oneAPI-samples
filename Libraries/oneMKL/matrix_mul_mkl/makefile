# Makefile for NMAKE

default: all

all: sgemm.exe dgemm.exe

run: sgemm.exe dgemm.exe
	.\sgemm.exe
	.\dgemm.exe

DPCPP_OPTS=/I"$(MKLROOT)\include" /Qmkl /EHsc -fsycl-device-code-split=per_kernel OpenCL.lib

sgemm.exe: matrix_mul_mkl.cpp
	icx-cl -fsycl matrix_mul_mkl.cpp /Fesgemm.exe $(DPCPP_OPTS)

dgemm.exe: matrix_mul_mkl.cpp
	icx-cl -fsycl matrix_mul_mkl.cpp /Fedgemm.exe $(DPCPP_OPTS) -DUSE_DOUBLE

clean:
	del /q sgemm.exe sgemm.exp sgemm.lib dgemm.exe dgemm.exp dgemm.lib

pseudo: clean run all
