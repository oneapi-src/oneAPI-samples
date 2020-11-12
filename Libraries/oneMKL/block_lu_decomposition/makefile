# Makefile for NMAKE

all: factor.exe solve.exe
	.\factor.exe
	.\solve.exe

DPCPP_OPTS=/I"$(MKLROOT)\include" /Qmkl /DMKL_ILP64 /EHsc -fsycl-device-code-split=per_kernel OpenCL.lib

factor.exe: factor.cpp dgeblttrf.cpp auxi.cpp
	dpcpp factor.cpp dgeblttrf.cpp auxi.cpp /Fefactor.exe $(DPCPP_OPTS)

solve.exe: solve.cpp dgeblttrf.cpp dgeblttrs.cpp auxi.cpp
	dpcpp solve.cpp dgeblttrf.cpp dgeblttrs.cpp auxi.cpp /Fesolve.exe $(DPCPP_OPTS)

clean:
	del /q factor.exe factor.exp factor.lib solve.exe solve.exp solve.lib
