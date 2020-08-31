# Makefile for GNU Make

default: run

run: sparse_cg
	./sparse_cg

DPCPP_OPTS = -I${MKLROOT}/include -mkl -fsycl-device-code-split=per_kernel

sparse_cg: sparse_cg.cpp
	dpcpp $< -o $@ $(DPCPP_OPTS)

clean:
	-rm -f sparse_cg

.PHONY: clean run
