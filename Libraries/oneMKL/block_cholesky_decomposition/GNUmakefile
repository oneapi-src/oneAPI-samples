# Makefile for GNU make

all: factor solve
	./factor
	./solve

MKL_COPTS = -DMKL_ILP64  -qmkl=sequential
MKL_LIBS  = -lsycl -lOpenCL -lpthread -lm -ldl

factor: factor.cpp dpbltrf.cpp auxi.cpp
	icpx $^ -o $@ -fsycl -fsycl-device-code-split=per_kernel $(MKL_COPTS) $(MKL_LIBS)

solve: solve.cpp dpbltrf.cpp dpbltrs.cpp auxi.cpp
	icpx $^ -o $@ -fsycl -fsycl-device-code-split=per_kernel $(MKL_COPTS) $(MKL_LIBS)

clean:
	-rm -f factor solve

.PHONY: all clean
