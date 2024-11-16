# Makefile for NMAKE

default: run

all: run

run: sparse_cg.exe sparse_cg2.exe
	.\sparse_cg
	.\sparse_cg2

SYCL_OPTS=/I"$(MKLROOT)\include" /Qmkl /Qmkl-sycl-impl="blas,sparse" /EHsc -fsycl-device-code-split=per_kernel OpenCL.lib

sparse_cg.exe: sparse_cg.cpp
	icx-cl -fsycl sparse_cg.cpp /Fesparse_cg.exe $(SYCL_OPTS)

sparse_cg2.exe: sparse_cg2.cpp
	icx-cl -fsycl sparse_cg2.cpp /Fesparse_cg2.exe $(SYCL_OPTS)

clean:
	del /q sparse_cg.exe sparse_cg.exp sparse_cg.lib
	del /q sparse_cg2.exe sparse_cg2.exp sparse_cg2.lib

pseudo: clean run all
