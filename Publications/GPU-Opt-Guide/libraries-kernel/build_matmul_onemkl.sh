icpx -fsycl -std=c++17 matmul_onemkl.cpp -o matmul_onemkl -DMKL_ILP64 -I${MKLROOT}/include \
      -lmkl_sycl -L${MKLROOT}/lib/intel64 \
      -lmkl_intel_ilp64 -lmkl_tbb_thread -lmkl_core -lsycl -lOpenCL -tbb -lpthread -lm -ldl
