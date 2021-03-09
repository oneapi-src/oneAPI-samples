# Makefile for GNU Make

default: run

all: run

run: sparse_cg
	./sparse_cg

DPCPP_OPTS = -qmkl -fsycl-device-code-split=per_kernel

sparse_cg: sparse_cg.cpp
	dpcpp $< -o $@ $(DPCPP_OPTS)

clean:
	-rm -f sparse_cg

.PHONY: clean run all
