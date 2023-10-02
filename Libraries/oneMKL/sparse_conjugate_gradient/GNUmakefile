# Makefile for GNU Make

default: run

all: run

run: sparse_cg
	./sparse_cg

MKL_COPTS = -DMKL_ILP64  -qmkl=sequential
MKL_LIBS  = -lsycl -lOpenCL -lpthread -lm -ldl

DPCPP_OPTS = $(MKL_COPTS) -fsycl-device-code-split=per_kernel $(MKL_LIBS)

sparse_cg: sparse_cg.cpp
	icpx $< -fsycl -o $@ $(DPCPP_OPTS)

clean:
	-rm -f sparse_cg genxir

.PHONY: clean run all
