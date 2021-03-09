# Makefile for GNU make

all: factor solve
	./factor
	./solve

factor: factor.cpp dgeblttrf.cpp auxi.cpp
	dpcpp $^ -o $@ -fsycl-device-code-split=per_kernel -DMKL_ILP64 -qmkl

solve: solve.cpp dgeblttrf.cpp dgeblttrs.cpp auxi.cpp
	dpcpp $^ -o $@ -fsycl-device-code-split=per_kernel -DMKL_ILP64 -qmkl

clean:
	-rm -f factor solve

.PHONY: clean all
