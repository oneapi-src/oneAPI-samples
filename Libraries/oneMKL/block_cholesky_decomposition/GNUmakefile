# Makefile for GNU make

all: factor solve
	./factor
	./solve

MKL_COPTS = -DMKL_ILP64  -qmkl -qmkl-sycl-impl="blas,lapack"

factor: factor.cpp dpbltrf.cpp auxi.cpp
	icpx $^ -o $@ -fsycl -fsycl-device-code-split=per_kernel $(MKL_COPTS)

solve: solve.cpp dpbltrf.cpp dpbltrs.cpp auxi.cpp
	icpx $^ -o $@ -fsycl -fsycl-device-code-split=per_kernel $(MKL_COPTS)

clean:
	-rm -f factor solve

.PHONY: all clean
