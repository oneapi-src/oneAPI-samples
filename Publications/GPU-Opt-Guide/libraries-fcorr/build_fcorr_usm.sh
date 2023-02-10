icpx -fsycl -O3 -std=c++17 -DMKL_ILP64 -I"${MKLROOT}/include" fcorr_1d_usm.cpp \
      -o fcorr_1d_usm \
      -L${MKLROOT}/lib/intel64 -lmkl_sycl \
      -lmkl_intel_ilp64 -lmkl_tbb_thread -lmkl_core -lsycl -lOpenCL -lpthread -lm -ldl
