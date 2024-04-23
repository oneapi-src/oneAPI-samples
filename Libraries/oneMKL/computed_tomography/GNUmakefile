#Makefile for GNU make

default: run

all: run

run: computed_tomography
	./computed_tomography 400 400 input.bmp radon.bmp restored.bmp

MKL_COPTS = -DMKL_ILP64  -qmkl=sequential
MKL_LIBS  = -lsycl -lOpenCL -lpthread -lm -ldl

DPCPP_OPTS = $(MKL_COPTS) -fsycl-device-code-split=per_kernel $(MKL_LIBS)

computed_tomography: computed_tomography.cpp
	icpx $< -fsycl -o $@ $(DPCPP_OPTS)

clean:
	-rm -f computed_tomography radon.bmp restored.bmp

.PHONY: clean run all
