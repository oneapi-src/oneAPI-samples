default: run_all

run_all: mc_european.exe mc_european_usm.exe
	.\mc_european
	.\mc_european_usm

DPCPP_OPTS=/I"$(MKLROOT)\include" /Qmkl /DMKL_ILP64 /EHsc -fsycl-device-code-split=per_kernel OpenCL.lib

mc_european.exe: mc_european.cpp
	dpcpp mc_european.cpp /Femc_european.exe $(DPCPP_OPTS)

mc_european_usm.exe: mc_european_usm.cpp
	dpcpp mc_european_usm.cpp /Femc_european_usm.exe $(DPCPP_OPTS)

clean:
	del /q mc_european.exe mc_european_usm.exe
	del /q mc_european.exp mc_european_usm.exp
	del /q mc_european.lib mc_european_usm.lib

pseudo: run_all clean

