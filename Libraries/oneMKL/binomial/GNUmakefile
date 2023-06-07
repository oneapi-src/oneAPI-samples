all: binomial_sycl

init_on_host ?= 0

MKL_COPTS = -DMKL_ILP64  -I"${MKLROOT}/include"
MKL_LIBS = -L${MKLROOT}/lib/intel64 -lmkl_sycl -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core

binomial_sycl: src/binomial_sycl.cpp src/binomial_main.cpp src/binomial.hpp
	icpx -fsycl -O3 -DSMALL_OPT_N=0 -DVERBOSE=1 -DREPORT_COLD=1 -DREPORT_WARM=1 -DINIT_ON_HOST=$(init_on_host)  $(MKL_COPTS) -o $@ src/binomial_main.cpp src/binomial_sycl.cpp $(MKL_LIBS)

clean:
	@rm -f binomial_sycl

.PHONY: clean all
