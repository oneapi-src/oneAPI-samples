all: binomial_sycl

init_on_host ?= 0

MKL_COPTS = -DMKL_ILP64  -qmkl -qmkl-sycl-impl=rng

binomial_sycl: src/binomial_sycl.cpp src/binomial_main.cpp src/binomial.hpp
	icpx -fsycl -O3 -DSMALL_OPT_N=0 -DVERBOSE=1 -DREPORT_COLD=1 -DREPORT_WARM=1 -DINIT_ON_HOST=$(init_on_host)  $(MKL_COPTS) -o $@ src/binomial_main.cpp src/binomial_sycl.cpp

clean:
	@rm -f binomial_sycl

.PHONY: clean all
