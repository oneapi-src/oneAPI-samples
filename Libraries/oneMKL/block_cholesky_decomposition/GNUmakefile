# Makefile for GNU make

all: factor solve
	./factor
	./solve

MKL_COPTS = -DMKL_ILP64  -I"${MKLROOT}/include"
MKL_LIBS = -L${MKLROOT}/lib/intel64 -lmkl_sycl -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lsycl -lOpenCL -lpthread -lm -ldl

factor: factor.cpp dpbltrf.cpp auxi.cpp
	dpcpp $^ -o $@ -fsycl-device-code-split=per_kernel $(MKL_COPTS) $(MKL_LIBS)

solve: solve.cpp dpbltrf.cpp dpbltrs.cpp auxi.cpp
	dpcpp $^ -o $@ -fsycl-device-code-split=per_kernel $(MKL_COPTS) $(MKL_LIBS)

clean:
	-rm -f factor solve

.PHONY: all clean
