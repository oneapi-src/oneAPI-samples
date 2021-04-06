# Makefile for GNU Make

default: run

all: run

run: sparse_cg
	./sparse_cg

MKL_COPTS = -DMKL_ILP64  -I"${MKLROOT}/include"
MKL_LIBS = -L${MKLROOT}/lib/intel64 -lmkl_sycl -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lsycl -lOpenCL -lpthread -lm -ldl

DPCPP_OPTS = $(MKL_COPTS) -fsycl-device-code-split=per_kernel $(MKL_LIBS)

sparse_cg: sparse_cg.cpp
	dpcpp $< -o $@ $(DPCPP_OPTS)

clean:
	-rm -f sparse_cg

.PHONY: clean run all
