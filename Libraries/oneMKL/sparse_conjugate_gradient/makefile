# Makefile for NMAKE

default: run

run: sparse_cg.exe
	.\sparse_cg

DPCPP_OPTS=/I"$(MKLROOT)\include" /Qmkl /EHsc -fsycl-device-code-split=per_kernel OpenCL.lib

sparse_cg.exe: sparse_cg.cpp
	dpcpp sparse_cg.cpp /Fesparse_cg.exe $(DPCPP_OPTS)

clean:
	del /q sparse_cg.exe sparse_cg.exp sparse_cg.lib

pseudo: clean run
