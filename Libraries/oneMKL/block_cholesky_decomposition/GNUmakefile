# Makefile for GNU make

all: factor solve
	./factor
	./solve

factor: factor.cpp dpbltrf.cpp auxi.cpp
	dpcpp $^ -o $@ -fsycl-device-code-split=per_kernel -DMKL_ILP64 -qmkl

solve: solve.cpp dpbltrf.cpp dpbltrs.cpp auxi.cpp
	dpcpp $^ -o $@ -fsycl-device-code-split=per_kernel -DMKL_ILP64 -qmkl

clean:
	-rm -f factor solve

.PHONY: all clean
