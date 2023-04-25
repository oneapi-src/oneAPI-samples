all: montecarlo

generator ?= mcg59

ifeq ($(generator), mrg)
	GENERATOR = /DUSE_MRG
endif

ifeq ($(generator), philox)
	GENERATOR = /DUSE_PHILOX
endif

ifneq ($(generator), $(filter $(generator),mrg philox mcg59))
 	$(error "You use unknown generator. Please, use mrg philox or mcg59 (default)")
endif

DPCPP_OPTS=/I$(MKLROOT)\include /DMKL_ILP64 $(GENERATOR) -fsycl -fsycl-device-code-split=per_kernel -qmkl

montecarlo: src/montecarlo_main.cpp
	icpx src/montecarlo_main.cpp /omontecarlo.exe $(DPCPP_OPTS)

clean:
	del /q montecarlo.exe

pseudo: clean all