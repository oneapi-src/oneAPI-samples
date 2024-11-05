# Makefile for GNU Make

default: run

all: run

run: t_test t_test_usm
	./t_test
	./t_test_usm

MKL_COPTS = -DMKL_ILP64  -qmkl -qmkl-sycl-impl="stats,rng"

DPCPP_OPTS = $(MKL_COPTS) -fsycl-device-code-split=per_kernel -fno-sycl-early-optimizations

t_test: t_test.cpp
	icpx $< -fsycl -o $@ $(DPCPP_OPTS)

t_test_usm: t_test_usm.cpp
	icpx $< -fsycl -o $@ $(DPCPP_OPTS)

clean:
	-rm -f t_test t_test_usm

.PHONY: clean run all
