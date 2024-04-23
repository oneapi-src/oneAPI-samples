icpx -fsycl -O3 -std=c++17 -DMKL_ILP64 -I"${MKLROOT}/include" rng_test.cpp \
      -o rng_test \
      -L${MKLROOT}/lib/intel64 -lmkl_sycl \
      -lmkl_intel_ilp64 -lmkl_tbb_thread -lmkl_core -lsycl -lOpenCL -lpthread -lm -ldl
