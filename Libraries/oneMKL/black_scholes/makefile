# ==============================================================================
#  Content:
#      Black-Scholes formula example makefile
# ==============================================================================

all: black_scholes_sycl.exe 


DPCPP_OPTS=-O3 /I$(MKLROOT)\include /DMKL_ILP64 /DVERBOSE=1 -fsycl -qmkl

black_scholes_sycl.exe: src\black_scholes_sycl.cpp
	icpx $(DPCPP_OPTS) src\black_scholes_sycl.cpp /oblack_scholes_sycl.exe

clean:
	del /q black_scholes_sycl.exe

.PHONY: clean all
