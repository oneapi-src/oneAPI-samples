# Makefile for GNU Make

default: run

all: run

run: mc_pi mc_pi_usm mc_pi_device_api
	./mc_pi
	./mc_pi_usm
	./mc_pi_device_api

MKL_COPTS = -DMKL_ILP64  -qmkl=sequential
MKL_LIBS  = -lsycl -lOpenCL -lpthread -lm -ldl

DPCPP_OPTS = $(MKL_COPTS) -fsycl-device-code-split=per_kernel -fno-sycl-early-optimizations $(MKL_LIBS)

mc_pi: mc_pi.cpp
	icpx $< -fsycl -o $@ $(DPCPP_OPTS)

mc_pi_usm: mc_pi_usm.cpp
	icpx $< -fsycl -o $@ $(DPCPP_OPTS)

mc_pi_device_api: mc_pi_device_api.cpp
	icpx $< -fsycl -o $@ $(DPCPP_OPTS)

clean:
	-rm -f mc_pi mc_pi_usm mc_pi_device_api

.PHONY: clean run all
