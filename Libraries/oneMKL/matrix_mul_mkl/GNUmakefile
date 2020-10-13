# Makefile for GNU Make

default: run

run: matrix_mul_mkl
	./matrix_mul_mkl

DPCPP_OPTS = -I${MKLROOT}/include -mkl -fsycl-device-code-split=per_kernel

matrix_mul_mkl: matrix_mul_mkl.cpp
	dpcpp $< -o $@ $(DPCPP_OPTS)

clean:
	-rm -f matrix_mul_mkl

.PHONY: clean run
