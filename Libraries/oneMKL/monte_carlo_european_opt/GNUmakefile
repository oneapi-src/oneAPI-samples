default: run_all

run_all: mc_european mc_european_usm
	./mc_european
	@echo
	./mc_european_usm

DPCPP_OPTS = -I${MKLROOT}/include -mkl -DMKL_ILP64 -fsycl-device-code-split=per_kernel

mc_european: mc_european.cpp
	dpcpp $< -o $@ $(DPCPP_OPTS)

mc_european_usm: mc_european_usm.cpp
	dpcpp $< -o $@ $(DPCPP_OPTS)

clean:
	-rm -f mc_european mc_european_usm

.PHONY: run_all clean
