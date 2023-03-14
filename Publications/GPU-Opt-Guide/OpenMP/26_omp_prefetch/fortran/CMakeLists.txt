add_compile_options(-fpconstant -fpp -ffast-math -fno-sycl-instrument-device-code)
add_fortran_example_with_mkl(nbody_f)
