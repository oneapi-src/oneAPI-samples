all: binomial_sycl.exe 

!if "$(init_on_host)" == "1"
	INIT_ON_HOST=/DINIT_ON_HOST=1
!endif

DPCPP_OPTS=-O3 /I$(MKLROOT)\include /DMKL_ILP64 /DVERBOSE=1 /DSMALL_OPT_N=0 /DREPORT_COLD=1 /DREPORT_WARM=1 $(INIT_ON_HOST) -fsycl -qmkl

binomial_sycl.exe: src\binomial_sycl.cpp src\binomial_main.cpp src\binomial.hpp
	icpx $(DPCPP_OPTS) /DVERBOSE=1 /DSMALL_OPT_N=0 /DREPORT_COLD=1 /DREPORT_WARM=1 src\binomial_sycl.cpp src\binomial_main.cpp /obinomial_sycl.exe

clean:
	del /q binomial_sycl.exe

.PHONY: clean all
