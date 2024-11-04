add_example(omp_interop_sycl_1)
add_example_with_mkl(omp_interop_sycl_2)
# Example omp_interop_sycl_3 with Fortran and C++ code
set(name "omp_interop_sycl_3")
add_executable(${name} ${name}a.f90 ${name}b.cpp)
target_compile_options(${name} PRIVATE -fiopenmp -fopenmp-targets=spir64 -fsycl)
set_target_properties(${name} PROPERTIES LINKER_LANGUAGE Fortran LINK_FLAGS -fsycl)
add_test(NAME ${name} COMMAND ${name})
