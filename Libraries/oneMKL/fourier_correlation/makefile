# Makefile for NMAKE

default: run_all

all: run_all

run_all: fcorr_1d_buff.exe fcorr_1d_usm.exe
	.\fcorr_1d_buff 1024
	.\fcorr_1d_usm 1024
	.\fcorr_2d_usm

DPCPP_OPTS=-DMKL_ILP64 -I"%MKLROOT%\include" /Qmkl OpenCL.lib /EHsc

fcorr_1d_buff.exe: fcorr_1d_buffers.cpp
	icx-cl -fsycl fcorr_1d_buffers.cpp /Fefcorr_1d_buff.exe $(DPCPP_OPTS)

fcorr_1d_usm.exe: fcorr_1d_usm.cpp
	icx-cl -fsycl fcorr_1d_usm.cpp /Fefcorr_1d_usm.exe $(DPCPP_OPTS)

fcorr_2d_usm.exe: fcorr_2d_usm.cpp
	icx-cl -fsycl fcorr_2d_usm.cpp /Fefcorr_2d_usm.exe $(DPCPP_OPTS)

clean:
	del /q fcorr_1d_buff.exe fcorr_1d_usm.exe fcorr_2d_usm.exe
	del /q fcorr_1d_buff.exp fcorr_1d_usm.exp fcorr_2d_usm.exp

pseudo: run_all clean all

