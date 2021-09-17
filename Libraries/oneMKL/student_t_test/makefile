# Makefile for NMAKE

default: run

all: run

run: t_test.exe t_test_usm.exe
	.\t_test.exe
	.\t_test_usm.exe

DPCPP_OPTS=/I"$(MKLROOT)\include" /Qmkl /DMKL_ILP64 /EHsc -fsycl-device-code-split=per_kernel -fno-sycl-early-optimizations OpenCL.lib

t_test.exe: t_test.cpp
	dpcpp t_test.cpp /Fet_test.exe $(DPCPP_OPTS)

t_test_usm.exe: t_test_usm.cpp
	dpcpp t_test_usm.cpp /Fet_test_usm.exe $(DPCPP_OPTS)

clean:
	del /q t_test.exe t_test_usm.exe

pseudo: clean run all
