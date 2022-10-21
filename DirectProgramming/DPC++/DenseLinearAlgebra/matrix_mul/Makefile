DPCPP_CXX = dpcpp
DPCPP_CXXFLAGS = -std=c++17 -g -o
DPCPP_LDFLAGS = 
DPCPP_EXE_NAME = matrix_mul_sycl
DPCPP_SOURCES = src/matrix_mul_sycl.cpp

CXX = icpx
OMP_CXXFLAGS = -fiopenmp -fopenmp-targets=spir64 -D__STRICT_ANSI__ -g -o
OMP_LDFLAGS = 
OMP_EXE_NAME = matrix_mul_omp
OMP_SOURCES = src/matrix_mul_omp.cpp

all:
	$(DPCPP_CXX) $(DPCPP_CXXFLAGS) $(DPCPP_EXE_NAME) $(DPCPP_SOURCES) $(DPCPP_LDFLAGS)

build_sycl:
	$(DPCPP_CXX) $(DPCPP_CXXFLAGS) $(DPCPP_EXE_NAME) $(DPCPP_SOURCES) $(DPCPP_LDFLAGS)

build_omp:
	$(CXX) $(OMP_CXXFLAGS) $(OMP_EXE_NAME) $(OMP_SOURCES) $(OMP_LDFLAGS)


run:
	./$(DPCPP_EXE_NAME)

run_sycl:
	./$(DPCPP_EXE_NAME)

run_omp:
	./$(OMP_EXE_NAME)


clean: 
	rm -rf $(DPCPP_EXE_NAME) $(OMP_EXE_NAME)



