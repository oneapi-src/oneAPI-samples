# Makefile for NMAKE

default: run_all

all: run_all

run_all: fcorr_1d_buff.exe fcorr_1d_usm.exe
	.\fcorr_1d_buff 1024
	.\fcorr_1d_usm 1024

DPCPP_OPTS=-DMKL_ILP64  -I"%MKLROOT%\include" mkl_sycl_dll.lib mkl_intel_ilp64_dll.lib mkl_tbb_thread_dll.lib mkl_core_dll.lib tbb.lib sycl.lib OpenCL.lib

fcorr_1d_buff.exe: fcorr_1d_buffers.cpp
	dpcpp fcorr_1d_buffers.cpp /Fefcorr_1d_buff.exe $(DPCPP_OPTS)

fcorr_1d_usm.exe: fcorr_1d_usm.cpp
	dpcpp fcorr_1d_usm.cpp /Fefcorr_1d_usm.exe $(DPCPP_OPTS)

clean:
	del /q fcorr_1d_buff.exe fcorr_1d_usm.exe
	del /q fcorr_1d_buff.exp fcorr_1d_usm.exp

pseudo: run_all clean all

