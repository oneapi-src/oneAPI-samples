# Makefile for GNU make

all: factor solve
	./factor
	./solve

factor: factor.cpp dpbltrf.cpp auxi.cpp
	dpcpp $^ -o $@ -fsycl-device-code-split=per_kernel -DMKL_ILP64 -I${MKLROOT}/include -mkl

solve: solve.cpp dpbltrf.cpp dpbltrs.cpp auxi.cpp
	dpcpp $^ -o $@ -fsycl-device-code-split=per_kernel -DMKL_ILP64 -I${MKLROOT}/include -mkl

clean:
	-rm -f factor solve

.PHONY: all clean
