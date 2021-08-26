# Makefile for GNU Make

default: run

all: run

run: t_test t_test_usm
	./t_test
	./t_test_usm

MKL_COPTS = -DMKL_ILP64  -I"${MKLROOT}/include"
MKL_LIBS = -L${MKLROOT}/lib/intel64 -lmkl_sycl -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lsycl -lOpenCL -lpthread -lm -ldl

DPCPP_OPTS = $(MKL_COPTS) -fsycl-device-code-split=per_kernel -fno-sycl-early-optimizations $(MKL_LIBS)

t_test: t_test.cpp
	dpcpp $< -o $@ $(DPCPP_OPTS)

t_test_usm: t_test_usm.cpp
	dpcpp $< -o $@ $(DPCPP_OPTS)

clean:
	-rm -f t_test t_test_usm

.PHONY: clean run all
