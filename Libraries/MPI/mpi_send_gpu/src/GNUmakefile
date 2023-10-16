example = mpi_send_gpu_omp mpi_send_gpu_sycl

INCLUDES =
LDFLAGS  =

CFLAGS   = -qopenmp -fopenmp-targets=spir64 -Wall -Wformat-security -Werror=format-security
CXXFLAGS = -fsycl -Wall -Wformat-security -Werror=format-security
# Use icx from DPC++ oneAPI toolkit to compile. Please source DPCPP's vars.sh before compilation.
CC       = mpiicx
CXX      = mpiicpx

all: CFLAGS += -O2
all: CXXFLAGS += -O2
all: $(example)

debug: CFLAGS += -O0 -g
debug: CXXFLAGS += -O0 -g
debug: $(example)

% : %.c
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)

% : %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)

clean:
	-rm -f $(example).o $(example)
