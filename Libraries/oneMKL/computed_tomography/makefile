# Makefile for NMAKE

default: run

run: computed_tomography.exe
	.\computed_tomography.exe 400 400 input.bmp radon.bmp restored.bmp

DPCPP_OPTS=/I"$(MKLROOT)\include" /Qmkl /EHsc -fsycl-device-code-split=per_kernel OpenCL.lib

computed_tomography.exe: computed_tomography.cpp
	dpcpp computed_tomography.cpp /Fecomputed_tomography.exe $(DPCPP_OPTS)

clean:
	del /q computed_tomography.exe computed_tomography.exp computed_tomography.lib

pseudo: clean run
