# ==============================================================================
#  Content:
#      Black-Scholes formula example makefile
# ==============================================================================

all: black_scholes_sycl 

init_on_host ?= 0

MKL_COPTS = -DMKL_ILP64 -qmkl=sequential
MKL_LIBS  = -lsycl -lOpenCL -lpthread -lm -ldl

black_scholes_sycl: src/black_scholes_sycl.cpp
	icpx -O3 -g -fsycl $(MKL_COPTS) $(MKL_LIBS) -DVERBOSE=1 -DSMALL_OPT_N=0 -DINIT_ON_HOST=$(init_on_host) -o $@  src/black_scholes_sycl.cpp

clean:
	@rm -f black_scholes_sycl

.PHONY: clean all
