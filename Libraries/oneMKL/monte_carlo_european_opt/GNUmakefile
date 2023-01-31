all: montecarlo

MKL_COPTS = -DMKL_ILP64  -I"${MKLROOT}/include"
MKL_LIBS = -L${MKLROOT}/lib/intel64 -lmkl_sycl -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl

DPCPP_OPTS = $(MKL_COPTS) -fsycl -fsycl-device-code-split=per_kernel $(MKL_LIBS)

montecarlo: src/montecarlo_main.cpp
	icpx $< -o $@ $(DPCPP_OPTS)

clean:
	-rm -f montecarlo

.PHONY: clean all