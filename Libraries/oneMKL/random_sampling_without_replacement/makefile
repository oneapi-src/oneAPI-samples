# Makefile for NMAKE

default: run

all: run

run: lottery.exe lottery_usm.exe lottery_device_api.exe
	.\lottery.exe
	.\lottery_usm.exe
	.\lottery_device_api.exe

DPCPP_OPTS=/I"$(MKLROOT)\include" /Qmkl /DMKL_ILP64 /EHsc -fsycl-device-code-split=per_kernel -fno-sycl-early-optimizations OpenCL.lib

lottery.exe: lottery.cpp
	dpcpp lottery.cpp /Felottery.exe $(DPCPP_OPTS)

lottery_usm.exe: lottery_usm.cpp
	dpcpp lottery_usm.cpp /Felottery_usm.exe $(DPCPP_OPTS)

lottery_device_api.exe: lottery_device_api.cpp
	dpcpp lottery_device_api.cpp /Felottery_device_api.exe $(DPCPP_OPTS)

clean:
	del /q lottery.exe lottery_usm.exe lottery_device_api.exe

pseudo: clean run all
