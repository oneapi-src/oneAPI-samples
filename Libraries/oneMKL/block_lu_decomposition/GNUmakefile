# Makefile for GNU make

all: factor solve
	./factor
	./solve

MKL_COPTS = -DMKL_ILP64  -qmkl=sequential
MKL_LIBS  = -lsycl -lOpenCL -lpthread -lm -ldl

factor: factor.cpp dgeblttrf.cpp auxi.cpp
	icpx $^ -o $@ -fsycl -fsycl-device-code-split=per_kernel $(MKL_COPTS) $(MKL_LIBS)

solve: solve.cpp dgeblttrf.cpp dgeblttrs.cpp auxi.cpp
	icpx $^ -o $@ -fsycl -fsycl-device-code-split=per_kernel $(MKL_COPTS) $(MKL_LIBS)

clean:
	-rm -f factor solve genxir

.PHONY: clean all
