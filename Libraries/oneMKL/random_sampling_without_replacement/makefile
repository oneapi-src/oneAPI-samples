# Makefile for NMAKE

default: run_all

run_all: lottery.exe lottery_usm.exe lottery_device_api.exe
	.\lottery
	.\lottery_usm
	.\lottery_device_api

DPCPP_OPTS=/I"$(MKLROOT)\include" /Qmkl /DMKL_ILP64 /EHsc -fsycl-device-code-split=per_kernel -fno-sycl-early-optimizations OpenCL.lib

lottery.exe: lottery.cpp
	dpcpp lottery.cpp /Felottery.exe $(DPCPP_OPTS)

lottery_usm.exe: lottery_usm.cpp
	dpcpp lottery_usm.cpp /Felottery_usm.exe $(DPCPP_OPTS)

lottery_device_api.exe: lottery_device_api.cpp
	dpcpp lottery_device_api.cpp /Felottery_device_api.exe $(DPCPP_OPTS)

clean:
	del /q lottery.exe lottery_usm.exe lottery_device_api.exe
	del /q lottery.exp lottery_usm.exp lottery_device_api.exp
	del /q lottery.lib lottery_usm.lib lottery_device_api.lib

pseudo: run_all clean