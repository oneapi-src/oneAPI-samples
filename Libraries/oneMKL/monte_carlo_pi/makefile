# Makefile for NMAKE

default: run_all

run_all: mc_pi.exe mc_pi_usm.exe mc_pi_device_api.exe
	.\mc_pi
	.\mc_pi_usm
	.\mc_pi_device_api

DPCPP_OPTS=/I"$(MKLROOT)\include" /Qmkl /DMKL_ILP64 /EHsc -fsycl-device-code-split=per_kernel -fno-sycl-early-optimizations OpenCL.lib

mc_pi.exe: mc_pi.cpp
	dpcpp mc_pi.cpp /Femc_pi.exe $(DPCPP_OPTS)

mc_pi_usm.exe: mc_pi_usm.cpp
	dpcpp mc_pi_usm.cpp /Femc_pi_usm.exe $(DPCPP_OPTS)

mc_pi_device_api.exe: mc_pi_device_api.cpp
	dpcpp mc_pi_device_api.cpp /Femc_pi_device_api.exe $(DPCPP_OPTS)

clean:
	del /q mc_pi.exe mc_pi_usm.exe mc_pi_device_api.exe
	del /q mc_pi.exp mc_pi_usm.exp mc_pi_device_api.exp
	del /q mc_pi.lib mc_pi_usm.lib mc_pi_device_api.lib

pseudo: run_all clean

