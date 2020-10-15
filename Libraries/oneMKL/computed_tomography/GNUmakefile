#Makefile for GNU make

default: run

run: computed_tomography
	./computed_tomography 400 400 input.bmp radon.bmp restored.bmp

DPCPP_OPTS = -I${MKLROOT}/include -mkl -fsycl-device-code-split=per_kernel

computed_tomography: computed_tomography.cpp
	dpcpp $< -o $@ $(DPCPP_OPTS)

clean:
	-rm -f computed_tomography radon.bmp restored.bmp

.PHONY: clean run
