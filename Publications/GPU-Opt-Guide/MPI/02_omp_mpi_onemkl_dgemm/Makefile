#OMP_AFFINITIZATION = 0 to ensure affinitization through MPI environment variables, and = 1 to use OpenMP to affinitize the MPI rank
OMP_AFFINITIZATION=0

CC=mpicxx
INCLUDE=-I$(MKLROOT)/include
LIB="$(MKLROOT)/lib"/libmkl_sycl.a -Wl,--start-group "$(MKLROOT)/lib"/libmkl_intel_lp64.a "$(MKLROOT)/lib"/libmkl_intel_thread.a "$(MKLROOT)/lib"/libmkl_core.a -Wl,--end-group -lsycl -lOpenCL -liomp5 -lpthread -ldl -lm -lstdc++
CFLAGS=-cxx=icpx -fiopenmp -fopenmp-targets=spir64 -fsycl -DOMP_AFFINITIZATION=$(OMP_AFFINITIZATION)
CFLAGS2=-cxx=icpx -fsycl-device-code-split=per_kernel -fiopenmp -fopenmp-targets=spir64 -fsycl

dgemm: dgemm.o Makefile
	$(CC) $(CFLAGS2) dgemm.o $(LIB) -o dgemm

dgemm.o: dgemm.cpp Makefile
	$(CC) $(CFLAGS) $(INCLUDE) -c dgemm.cpp -o dgemm.o

clean:
	rm -rf ./dgemm ./dgemm.o
