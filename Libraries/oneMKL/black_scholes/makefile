all: binomial_sycl.exe 

!if "$(init_on_host)" == "1"
	INIT_ON_HOST=/DINIT_ON_HOST=1
!endif

DPCPP_OPTS=-O3 /I$(MKLROOT)\include /DMKL_ILP64 /DVERBOSE=1 /DSMALL_OPT_N=0 $(INIT_ON_HOST) -fsycl -qmkl

binomial_sycl.exe: src\binomial_sycl.cpp
	icpx $(DPCPP_OPTS) src\binomial_sycl.cpp /obinomial_sycl.exe

clean:
	del /q binomial_sycl.exe

.PHONY: clean all
