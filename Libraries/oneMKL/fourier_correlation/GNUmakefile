# Makefile for GNU Make

default: run_all

all: run_all

run_all: fcorr_1d_buff fcorr_1d_usm
	./fcorr_1d_buff 4096
	./fcorr_1d_usm 4096

DPCPP_OPTS = -DMKL_ILP64 -I${MKLROOT}/include -L${MKLROOT}/lib/intel64 -lmkl_sycl -lmkl_intel_ilp64 -lmkl_tbb_thread -lmkl_core -lsycl -lOpenCL -lpthread -lm -ldl

fcorr_1d_buff: fcorr_1d_buffers.cpp
	dpcpp $< -o $@ $(DPCPP_OPTS)

fcorr_1d_usm: fcorr_1d_usm.cpp
	dpcpp $< -o $@ $(DPCPP_OPTS)

clean:
	-rm -f fcorr_1d_buff fcorr_1d_usm

.PHONY: run_all clean all
