# Makefile for GNU Make

default: run_all

run_all: lottery lottery_usm lottery_device_api
		./lottery
		./lottery_usm
		./lottery_device_api

DPCPP_OPTS = -I${MKLROOT}/include -mkl -DMKL_ILP64 -fsycl-device-code-split=per_kernel -fno-sycl-early-optimizations

lottery: lottery.cpp
		dpcpp $< -o $@ $(DPCPP_OPTS)

lottery_usm: lottery_usm.cpp
		dpcpp $< -o $@ $(DPCPP_OPTS)

lottery_device_api: lottery_device_api.cpp
		dpcpp $< -o $@ $(DPCPP_OPTS)

clean:
		-rm -f lottery lottery_usm lottery_device_api

.PHONY: run_all clean