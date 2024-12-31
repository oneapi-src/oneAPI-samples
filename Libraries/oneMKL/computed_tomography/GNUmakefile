#Makefile for GNU make

default: run

all: run

run: computed_tomography
	./computed_tomography

MKL_COPTS = -DMKL_ILP64  -qmkl -qmkl-sycl-impl=dft

DPCPP_OPTS = $(MKL_COPTS) -fsycl-device-code-split=per_kernel

computed_tomography: computed_tomography.cpp
	icpx $< -fsycl -o $@ $(DPCPP_OPTS)

clean:
	-rm -f computed_tomography radon.bmp restored.bmp

.PHONY: clean run all
