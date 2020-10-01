# Makefile for GNU Make

default: run_all

run_all: mc_pi mc_pi_usm mc_pi_device_api
	./mc_pi
	./mc_pi_usm
	./mc_pi_device_api

DPCPP_OPTS = -I${MKLROOT}/include -mkl -DMKL_ILP64 -fsycl-device-code-split=per_kernel -fno-sycl-early-optimizations

mc_pi: mc_pi.cpp
	dpcpp $< -o $@ $(DPCPP_OPTS)

mc_pi_usm: mc_pi_usm.cpp
	dpcpp $< -o $@ $(DPCPP_OPTS)

mc_pi_device_api: mc_pi_device_api.cpp
	dpcpp $< -o $@ $(DPCPP_OPTS)

clean:
	-rm -f mc_pi mc_pi_usm mc_pi_device_api

.PHONY: run_all clean
