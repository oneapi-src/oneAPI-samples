#Makefile for GNU make

default: run

all: run

run: computed_tomography
	./computed_tomography 400 400 input.bmp radon.bmp restored.bmp

MKL_COPTS = -DMKL_ILP64  -I"${MKLROOT}/include"
MKL_LIBS = -L${MKLROOT}/lib/intel64 -lmkl_sycl -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lsycl -lOpenCL -lpthread -lm -ldl

DPCPP_OPTS = $(MKL_COPTS) -fsycl-device-code-split=per_kernel $(MKL_LIBS)

computed_tomography: computed_tomography.cpp
	dpcpp $< -o $@ $(DPCPP_OPTS)

clean:
	-rm -f computed_tomography radon.bmp restored.bmp

.PHONY: clean run all
