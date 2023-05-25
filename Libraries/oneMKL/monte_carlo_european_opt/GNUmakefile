all: montecarlo

# setting non-default generator
generator ?= mcg59

ifeq ($(generator), mrg)
	GENERATOR = -DUSE_MRG
endif

ifeq ($(generator), philox)
	GENERATOR = -DUSE_PHILOX
endif

ifneq ($(generator), $(filter $(generator),mrg philox mcg59))
 	$(error "You use unknown generator. Please, use mrg philox or mcg59 (default)")
endif

# setting initial random number generation on host
init_on_host ?= 0

MKL_COPTS = -DMKL_ILP64 $(GENERATOR) -I"${MKLROOT}/include"
MKL_LIBS = -L${MKLROOT}/lib/intel64 -lmkl_sycl -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core

DPCPP_OPTS = $(MKL_COPTS) -fsycl -DINIT_ON_HOST=$(init_on_host) $(MKL_LIBS)

montecarlo: src/montecarlo_main.cpp
	icpx $< -o $@ $(DPCPP_OPTS)

clean:
	-rm -f montecarlo

.PHONY: clean all