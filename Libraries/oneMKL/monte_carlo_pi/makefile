# Makefile for NMAKE

default: run

all: run

run: mc_pi.exe mc_pi_usm.exe mc_pi_device_api.exe
	.\mc_pi.exe
	.\mc_pi_usm.exe
	.\mc_pi_device_api.exe

DPCPP_OPTS=/I"$(MKLROOT)\include" /Qmkl /DMKL_ILP64 /EHsc -fsycl-device-code-split=per_kernel -fno-sycl-early-optimizations OpenCL.lib

mc_pi.exe: mc_pi.cpp
	dpcpp mc_pi.cpp /Femc_pi.exe $(DPCPP_OPTS)

mc_pi_usm.exe: mc_pi_usm.cpp
	dpcpp mc_pi_usm.cpp /Femc_pi_usm.exe $(DPCPP_OPTS)

mc_pi_device_api.exe: mc_pi_device_api.cpp
	dpcpp mc_pi_device_api.cpp /Femc_pi_device_api.exe $(DPCPP_OPTS)

clean:
	del /q mc_pi.exe mc_pi_usm.exe mc_pi_device_api.exe

pseudo: clean run all
