
CXX = icpx

SYCL_CXXFLAGS = -std=c++17 -fsycl -g -o
SYCL_LDFLAGS = 
SYCL_EXE_NAME = matrix_mul_dpc
SYCL_SOURCES = src/matrix_mul_sycl.cpp

OMP_CXXFLAGS = -fiopenmp -fopenmp-targets=spir64 -D__STRICT_ANSI__ -g -o
OMP_LDFLAGS = 
OMP_EXE_NAME = matrix_mul_omp
OMP_SOURCES = src/matrix_mul_omp.cpp

all:
	$(CXX) $(SYCL_CXXFLAGS) $(SYCL_EXE_NAME) $(SYCL_SOURCES) $(SYCL_LDFLAGS)

build_sycl:
	$(CXX) $(SYCL_CXXFLAGS) $(SYCL_EXE_NAME) $(SYCL_SOURCES) $(SYCL_LDFLAGS)

build_omp:
	$(CXX) $(OMP_CXXFLAGS) $(OMP_EXE_NAME) $(OMP_SOURCES) $(OMP_LDFLAGS)


run:
	./$(SYCL_EXE_NAME)

run_sycl:
	./$(SYCL_EXE_NAME)

run_omp:
	./$(OMP_EXE_NAME)


clean: 
	rm -rf $(SYCL_EXE_NAME) $(OMP_EXE_NAME)



