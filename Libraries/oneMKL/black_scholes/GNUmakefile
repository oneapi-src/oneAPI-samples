# ==============================================================================
#  Content:
#      Black-Scholes formula example makefile
# ==============================================================================

default: all

all: black_scholes.run

N          ?= 10000000
ACC        ?=

mkl_path   := $(MKL)
acc        := $(strip $(ACC))
headers    := $(wildcard *.hpp)

cxx_flags  := -O3 -DMKL_ILP64 -I$(MKLROOT)/include -fno-sycl-early-optimizations
ldxx_flags := -mkl -fsycl-device-code-split=per_kernel
cxx        := dpcpp

ifneq ($(acc),)
cxx_flags += -DACC_$(acc)
endif

black_scholes: black_scholes.cpp $(headers)
	$(cxx) $(cxx_flags) $< $(ldxx_flags) -o $@

black_scholes.run: black_scholes
	./$< $(N)

.PHONY: clean help black_scholes.run

clean:
	rm -f black_scholes

help:
	@echo "Black Scholes oneAPI MKL VM sample"
	@echo "make [nopt] [ACC=la|ha|ep]"
	@echo ""
	@echo "ACC defines the accuracy:"
	@echo "   ha: high accuracy (most accurate)"
	@echo "   la: low accuracy"
	@echo "   ep: extended performance (fastest)"

