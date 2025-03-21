# Makefile for GNU make

all: factor solve
	./factor
	./solve

MKL_COPTS = -DMKL_ILP64  -qmkl -qmkl-sycl-impl="blas,lapack"

factor: factor.cpp dgeblttrf.cpp auxi.cpp
	icpx $^ -o $@ -fsycl -fsycl-device-code-split=per_kernel $(MKL_COPTS)

solve: solve.cpp dgeblttrf.cpp dgeblttrs.cpp auxi.cpp
	icpx $^ -o $@ -fsycl -fsycl-device-code-split=per_kernel $(MKL_COPTS)

clean:
	-rm -f factor solve genxir

.PHONY: clean all
