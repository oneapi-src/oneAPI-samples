default: run

all: run

run: mc_european mc_european_usm
	./mc_european
	@echo
	./mc_european_usm

MKL_COPTS = -DMKL_ILP64  -I"${MKLROOT}/include"
MKL_LIBS = -L${MKLROOT}/lib/intel64 -lmkl_sycl -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lsycl -lOpenCL -lpthread -lm -ldl

DPCPP_OPTS = $(MKL_COPTS) -fsycl-device-code-split=per_kernel $(MKL_LIBS)

mc_european: mc_european.cpp
	dpcpp $< -o $@ $(DPCPP_OPTS)

mc_european_usm: mc_european_usm.cpp
	dpcpp $< -o $@ $(DPCPP_OPTS)

clean:
	-rm -f mc_european mc_european_usm

.PHONY: clean run all
