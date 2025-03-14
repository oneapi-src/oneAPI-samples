# Makefile for NMAKE

default: run

all: run

run: computed_tomography.exe
	.\computed_tomography.exe

DPCPP_OPTS=/I"$(MKLROOT)\include" /Qmkl /Qmkl-sycl-impl=dft /EHsc -fsycl-device-code-split=per_kernel OpenCL.lib

computed_tomography.exe: computed_tomography.cpp
	icx-cl -fsycl $< /Fe$@ $(DPCPP_OPTS)

clean:
	del /q /f computed_tomography.exe computed_tomography.exp computed_tomography.lib radon.bmp restored.bmp errors.bmp

pseudo: clean run all
